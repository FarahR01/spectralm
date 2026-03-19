"""
SpectraLM training script.

Usage:
    python scripts/train.py --data_dir data/processed --epochs 60 --batch_size 32

Design notes:
    - Mixed precision (bfloat16) for speed on modern GPUs
    - Gradient clipping to handle physics loss spikes in early training
    - Physics loss is annealed in: λ starts at 0 and reaches its full value
      at epoch 10. This prevents the constraint from overwhelming the CE loss
      before the model has learned basic SMILES syntax.
    - Rich console progress with per-batch ECR monitoring
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from spectralm.models import SpectraLM, SpectraLMConfig
from spectralm.models.transformer import PAD_IDX, smiles_tokenise
from spectralm.data.augmentation import SpectralAugmentor, AugmentationConfig

console = Console()


# ── Dataset ───────────────────────────────────────────────────────────────────
class IRSpectraDataset(Dataset):
    """
    Loads pre-processed IR spectra and SMILES labels.

    Expected data format (from data/processed/):
        spectra.pt  : torch.Tensor (N, 1800) — normalised absorbance
        smiles.json : list of N SMILES strings
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        augment: bool = True,
        max_smiles_len: int = 128,
    ):
        self.max_len = max_smiles_len
        spectra_path = data_dir / f"{split}_spectra.pt"
        smiles_path = data_dir / f"{split}_smiles.json"

        self.spectra = torch.load(spectra_path, weights_only=True)  # (N, 1800)
        with open(smiles_path) as f:
            self.smiles_list = json.load(f)

        assert len(self.spectra) == len(self.smiles_list), "Spectra/SMILES count mismatch"

        self.augmentor = SpectralAugmentor(AugmentationConfig()) if augment else None

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spectrum = self.spectra[idx].float()

        if self.augmentor is not None:
            spectrum = self.augmentor(spectrum)

        tokens = smiles_tokenise(self.smiles_list[idx])
        tokens = tokens[:self.max_len]  # Truncate if needed

        # Pad to max_len
        pad_len = self.max_len - len(tokens)
        tokens_tensor = torch.tensor(tokens + [PAD_IDX] * pad_len, dtype=torch.long)
        padding_mask = torch.tensor(
            [False] * len(tokens) + [True] * pad_len, dtype=torch.bool
        )

        return {
            "spectrum": spectrum,
            "tokens": tokens_tensor,
            "padding_mask": padding_mask,
        }


# ── Loss function ─────────────────────────────────────────────────────────────
def compute_loss(
    model: SpectraLM,
    batch: dict[str, torch.Tensor],
    physics_scale: float = 1.0,  # Annealing multiplier
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = CE_loss + λ_scale * (λ_bl * BL_ECR + λ_gf * GF_penalty)

    Teacher forcing: decoder input = tokens[:-1], target = tokens[1:]
    """
    spectrum = batch["spectrum"]
    tokens = batch["tokens"]
    padding_mask = batch["padding_mask"]

    tgt_in  = tokens[:, :-1]   # (B, T-1) — decoder input
    tgt_out = tokens[:, 1:]    # (B, T-1) — target (shifted)
    pad_mask_in = padding_mask[:, :-1]

    out = model(spectrum, tgt_in, pad_mask_in)

    # ── Cross-entropy on SMILES tokens ────────────────────────────────────
    logits = out["logits"]  # (B, T-1, V)
    B, T, V = logits.shape
    ce_loss = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_out.reshape(B * T),
        ignore_index=PAD_IDX,
    )

    # ── Physics losses (annealed) ──────────────────────────────────────────
    cfg = model.config
    bl_loss = out["bl_loss"] * cfg.lambda_beer_lambert * physics_scale
    gf_loss = out["gf_loss"] * cfg.lambda_group_freq * physics_scale

    total_loss = ce_loss + bl_loss + gf_loss

    metrics = {
        "loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "bl_loss": out["bl_loss"].item(),
        "gf_loss": out["gf_loss"].item(),
        "ecr_mean": out["bl_report"].mean_ecr,
        "implausible_rate": out["bl_report"].implausible_rate,
    }

    return total_loss, metrics


# ── Physics loss annealing schedule ──────────────────────────────────────────
def physics_lambda_schedule(epoch: int, warmup_epochs: int = 10) -> float:
    """
    Linearly anneal physics loss weight from 0 to 1 over warmup_epochs.
    This allows the model to first learn basic SMILES syntax before being
    penalised for physics violations.

    Key decision from experiments: using the full λ from epoch 0 caused
    the model to collapse to predicting only the most common molecules
    (those with the simplest spectra to reconstruct). Annealing fixed this.
    """
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace, config: SpectraLMConfig | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_ds = IRSpectraDataset(data_dir, split="train", augment=True)
    val_ds   = IRSpectraDataset(data_dir, split="val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4
    )
    console.print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    if config is None:
        config = SpectraLMConfig()
    model = SpectraLM(config).to(device)
    console.print(f"Parameters: {model.num_parameters:,}")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98)
    )
    # Cosine schedule with linear warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5  # 5 epoch warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_ecr = float("inf")
    history: list[dict] = []

    # ── Training ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        physics_scale = physics_lambda_schedule(epoch, warmup_epochs=10)
        model.train()
        train_metrics: list[dict] = []

        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
            console=console, transient=True
        ) as progress:
            task = progress.add_task(f"Epoch {epoch}/{args.epochs}", total=len(train_loader))

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimiser.zero_grad()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, metrics = compute_loss(model, batch, physics_scale)

                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()

                train_metrics.append(metrics)
                progress.advance(task)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_metrics: list[dict] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    _, metrics = compute_loss(model, batch, physics_scale=1.0)
                val_metrics.append(metrics)

        # ── Aggregate and log ──────────────────────────────────────────────
        def avg(lst, key):
            return sum(d[key] for d in lst) / len(lst)

        epoch_log = {
            "epoch": epoch,
            "train_loss":         avg(train_metrics, "loss"),
            "train_ce":           avg(train_metrics, "ce_loss"),
            "train_ecr":          avg(train_metrics, "ecr_mean"),
            "train_implausible":  avg(train_metrics, "implausible_rate"),
            "val_loss":           avg(val_metrics, "loss"),
            "val_ecr":            avg(val_metrics, "ecr_mean"),
            "val_implausible":    avg(val_metrics, "implausible_rate"),
            "physics_scale":      physics_scale,
            "lr":                 scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)

        # Rich table output
        table = Table(show_header=True, header_style="bold")
        for col in ["epoch", "train_loss", "val_loss", "val_ecr", "val_implausible", "physics_scale"]:
            table.add_column(col, justify="right")
        table.add_row(
            str(epoch),
            f"{epoch_log['train_loss']:.4f}",
            f"{epoch_log['val_loss']:.4f}",
            f"{epoch_log['val_ecr']:.4f}",
            f"{epoch_log['val_implausible']:.3%}",
            f"{physics_scale:.2f}",
        )
        console.print(table)

        # ── Save checkpoint ────────────────────────────────────────────────
        if epoch_log["val_ecr"] < best_val_ecr:
            best_val_ecr = epoch_log["val_ecr"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "config": None,
                "val_ecr": best_val_ecr,
                "history": history,
            }, ckpt_dir / "best_model.pt")
            console.print(f"[green]✓ New best ECR: {best_val_ecr:.4f}[/green]")

        # Always save latest
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   ckpt_dir / "latest.pt")

        # Save training history
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    console.print(f"\n[bold green]Training complete. Best val ECR: {best_val_ecr:.4f}[/bold green]")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpectraLM")
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()
    train(args)

