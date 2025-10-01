"""
evals/ablation_runner.py

Systematic ablation study runner.

Trains (or loads) each ablation variant and scores it with
DomainResidualScorer. Produces the ablation table from the paper
(Notebook 03, Table 1) as a reproducible artefact.

Each variant is defined as a config delta applied to the full
SpectraLMConfig. Results are cached to disk so partial runs
can be resumed without retraining.

Usage:
    # Run all ablations from scratch (slow — 60 epochs × 8 variants)
    python evals/ablation_runner.py --data_dir data/processed --epochs 60

    # Run quick validation (5 epochs per variant, for CI)
    python evals/ablation_runner.py --data_dir data/processed --epochs 5 --quick

    # Load cached results and regenerate table only
    python evals/ablation_runner.py --results_dir evals/results --table_only
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectralm.models import SpectraLM, SpectraLMConfig
from spectralm.models.encoder import EncoderConfig
from spectralm.models.transformer import TransformerConfig
from spectralm.scripts.train import IRSpectraDataset, train
from evals.domain_residuals import DomainResidualScorer


# ══════════════════════════════════════════════════════════════════════════════
# Ablation variant definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationVariant:
    """
    A single ablation: a name, a config modifier function, and a description
    of what was changed and why it tests a specific hypothesis.
    """
    name: str
    short_name: str           # For table display (≤ 38 chars)
    description: str          # What was changed
    hypothesis_tested: str    # Which hypothesis from 02_experiments this tests
    config_fn: Callable[[SpectraLMConfig], SpectraLMConfig]  # Config mutator

    # Populated after evaluation
    result: dict = field(default_factory=dict)
    trained: bool = False


def _make_variants() -> list[AblationVariant]:
    """
    Returns all ablation variants as AblationVariant objects.

    Each variant modifies exactly one aspect of the full SpectraLMConfig.
    This is the "one factor at a time" ablation strategy — each result
    is interpretable in isolation.
    """

    def no_physics(cfg: SpectraLMConfig) -> SpectraLMConfig:
        cfg.lambda_beer_lambert = 0.0
        cfg.lambda_group_freq   = 0.0
        return cfg

    def discrete_pe(cfg: SpectraLMConfig) -> SpectraLMConfig:
        # Disabling continuous PE requires modifying the encoder config.
        # We signal this via a custom attribute the encoder checks.
        cfg.encoder.use_continuous_pe = False
        return cfg

    def no_cnn_encoder(cfg: SpectraLMConfig) -> SpectraLMConfig:
        # Replace CNN with a single linear projection
        cfg.encoder.cnn_channels = [1, 256]
        cfg.encoder.cnn_kernels  = [1]    # kernel_size=1 → pointwise projection
        return cfg

    def no_group_freq_penalty(cfg: SpectraLMConfig) -> SpectraLMConfig:
        cfg.lambda_group_freq = 0.0
        return cfg

    def reduced_lambda(cfg: SpectraLMConfig) -> SpectraLMConfig:
        cfg.lambda_beer_lambert = 0.10
        return cfg

    def increased_lambda(cfg: SpectraLMConfig) -> SpectraLMConfig:
        cfg.lambda_beer_lambert = 0.50
        return cfg

    def no_augmentation(cfg: SpectraLMConfig) -> SpectraLMConfig:
        cfg.encoder._no_augmentation = True  # flag read by train()
        return cfg

    def full_model(cfg: SpectraLMConfig) -> SpectraLMConfig:
        return cfg  # no changes — this is the baseline

    return [
        AblationVariant(
            name="no_physics_loss",
            short_name="No physics loss (λ=0)",
            description="Both physics losses set to 0. Tests whether physics constraints add value.",
            hypothesis_tested="H5",
            config_fn=no_physics,
        ),
        AblationVariant(
            name="discrete_pe",
            short_name="No continuous PE (discrete)",
            description="Standard token-index PE replacing wavenumber PE.",
            hypothesis_tested="H4",
            config_fn=discrete_pe,
        ),
        AblationVariant(
            name="no_cnn_encoder",
            short_name="No CNN encoder (linear proj)",
            description="CNN encoder replaced with pointwise linear projection.",
            hypothesis_tested="H3",
            config_fn=no_cnn_encoder,
        ),
        AblationVariant(
            name="no_group_freq_penalty",
            short_name="No group freq penalty",
            description="Group frequency penalty removed, BL constraint kept.",
            hypothesis_tested="H5",
            config_fn=no_group_freq_penalty,
        ),
        AblationVariant(
            name="reduced_lambda_0_10",
            short_name="Reduced λ_bl = 0.10",
            description="BL constraint weight reduced from 0.30 to 0.10.",
            hypothesis_tested="H5 (λ sensitivity)",
            config_fn=reduced_lambda,
        ),
        AblationVariant(
            name="increased_lambda_0_50",
            short_name="Increased λ_bl = 0.50",
            description="BL constraint weight increased from 0.30 to 0.50.",
            hypothesis_tested="H5 (λ sensitivity)",
            config_fn=increased_lambda,
        ),
        AblationVariant(
            name="no_augmentation",
            short_name="No data augmentation",
            description="Training without spectral augmentation.",
            hypothesis_tested="Augmentation necessity",
            config_fn=no_augmentation,
        ),
        AblationVariant(
            name="full_model",
            short_name="Full SpectraLM (λ=0.30) ◀",
            description="Full model with all components. Reference row.",
            hypothesis_tested="All",
            config_fn=full_model,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

class AblationRunner:
    """
    Trains and evaluates each ablation variant, caching results to disk.

    For each variant:
      1. Construct config via config_fn
      2. Train for `epochs` epochs (or load cached checkpoint)
      3. Evaluate with DomainResidualScorer
      4. Cache result JSON to results_dir/

    The final ablation table is printed with rich and saved to
    results_dir/ablation_table.json.
    """

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        epochs: int = 60,
        batch_size: int = 32,
        device: str = "cuda",
        quick: bool = False,
    ):
        self.data_dir    = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.epochs     = 5 if quick else epochs
        self.batch_size = batch_size
        self.device     = device
        self.quick      = quick
        self.variants   = _make_variants()

        # Pre-load test data (shared across all variants)
        self.test_ds = IRSpectraDataset(self.data_dir, split="test", augment=False)
        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        import json
        test_smiles_path = self.data_dir / "test_smiles.json"
        with open(test_smiles_path) as f:
            self.test_smiles = json.load(f)

        console.print(f"[bold]Test set:[/bold] {len(self.test_ds):,} samples")
        console.print(f"[bold]Epochs:[/bold] {self.epochs} {'(quick mode)' if quick else ''}")

    def run(self, skip_training: bool = False) -> list[AblationVariant]:
        """Run all ablation variants."""
        for variant in track(self.variants, description="Running ablations"):
            cache_path = self.results_dir / f"{variant.name}.json"

            # Load cached result if available
            if cache_path.exists():
                with open(cache_path) as f:
                    variant.result = json.load(f)
                variant.trained = True
                console.print(f"  [green]Loaded cache:[/green] {variant.short_name}")
                continue

            console.print(f"\n[bold cyan]Running:[/bold cyan] {variant.short_name}")
            console.print(f"  Hypothesis tested: {variant.hypothesis_tested}")

            # ── Build config ────────────────────────────────────────────
            base_config = SpectraLMConfig()
            variant_config = variant.config_fn(deepcopy(base_config))

            # ── Train (or skip) ─────────────────────────────────────────
            ckpt_path = self.results_dir / f"{variant.name}_ckpt.pt"

            if not skip_training:
                ckpt_path = self._train_variant(variant, variant_config, ckpt_path)

            if not ckpt_path.exists():
                console.print(f"  [red]Checkpoint not found — skipping[/red]")
                continue

            # ── Evaluate ────────────────────────────────────────────────
            model = SpectraLM(variant_config).to(self.device)
            ckpt  = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            scorer = DomainResidualScorer(
                model, device=self.device, verbose=False
            )
            report = scorer.evaluate(
                self.test_loader, self.test_smiles,
                model_name=variant.name,
                max_samples=200 if self.quick else None,
            )

            variant.result = {
                "bleu_4":          report.bleu_4,
                "ecr_mean":        report.ecr_mean,
                "tanimoto_mean":   report.tanimoto_mean,
                "gf_recall_mean":  report.gf_recall_mean,
                "implausible_rate": report.implausible_rate,
                "valid_smiles_rate": report.valid_smiles_rate,
                "formula_accuracy": report.formula_accuracy,
                "eval_time_sec":   report.eval_time_sec,
            }
            variant.trained = True

            # Cache result
            with open(cache_path, "w") as f:
                json.dump({
                    "variant": variant.name,
                    "description": variant.description,
                    "hypothesis_tested": variant.hypothesis_tested,
                    **variant.result,
                }, f, indent=2)

            console.print(f"  [green]✓[/green] BLEU={report.bleu_4:.3f}"
                          f"  ECR={report.ecr_mean:.3f}"
                          f"  implaus={report.implausible_rate:.1%}")

        return self.variants

    def _train_variant(
        self,
        variant: AblationVariant,
        config: SpectraLMConfig,
        ckpt_path: Path,
    ) -> Path:
        """Train a single ablation variant and save checkpoint."""
        import argparse
        args = argparse.Namespace(
            data_dir=str(self.data_dir),
            ckpt_dir=str(ckpt_path.parent / variant.name),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=3e-4,
        )
        # Monkey-patch the config factory for this variant
        import spectralm.models as sm_models
        original_config = sm_models.SpectraLMConfig
        sm_models.SpectraLMConfig = lambda: config  # type: ignore

        try:
            train(args)
        finally:
            sm_models.SpectraLMConfig = original_config  # restore

        trained_ckpt = Path(args.ckpt_dir) / "best_model.pt"
        return trained_ckpt

    def print_table(self):
        """Print the ablation table with rich formatting."""
        table = Table(title="Ablation Study — Table 1", show_header=True,
                      header_style="bold", show_lines=True)

        table.add_column("Variant",          style="", width=38)
        table.add_column("BLEU-4",           justify="right", width=7)
        table.add_column("ECR",              justify="right", width=7)
        table.add_column("Tanimoto",         justify="right", width=9)
        table.add_column("GF Recall",        justify="right", width=9)
        table.add_column("Implaus %",        justify="right", width=9)

        full_model_idx = len(self.variants) - 1

        for i, variant in enumerate(self.variants):
            if not variant.result:
                continue
            r = variant.result
            is_full = (i == full_model_idx)
            style = "bold green" if is_full else ""

            table.add_row(
                variant.short_name,
                f"{r.get('bleu_4', 0):.3f}",
                f"{r.get('ecr_mean', 0):.3f}",
                f"{r.get('tanimoto_mean', 0):.3f}",
                f"{r.get('gf_recall_mean', 0):.3f}",
                f"{r.get('implausible_rate', 0):.1%}",
                style=style,
            )

        console.print(table)

    def save_table(self, path: Path | None = None):
        """Save ablation table to JSON."""
        path = path or self.results_dir / "ablation_table.json"
        rows = []
        for v in self.variants:
            rows.append({
                "name": v.name,
                "short_name": v.short_name,
                "hypothesis_tested": v.hypothesis_tested,
                **v.result,
            })
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        console.print(f"[green]Ablation table saved →[/green] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical significance testing
# ══════════════════════════════════════════════════════════════════════════════

class SignificanceTester:
    """
    Bootstrap confidence intervals and paired t-tests for ablation comparisons.

    Reports whether the difference between the full model and each ablation
    variant is statistically significant — a key requirement for rigorous
    ablation studies.
    """

    def __init__(self, full_model_samples: list[float], n_bootstrap: int = 10_000):
        """
        full_model_samples : list of per-sample ECR values for the full model
        """
        self.full_samples = np.array(full_model_samples)
        self.n_bootstrap  = n_bootstrap

    def bootstrap_ci(
        self, samples: np.ndarray, metric_fn: Callable = np.mean, ci: float = 0.95
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for a metric."""
        bootstrapped = []
        for _ in range(self.n_bootstrap):
            resampled = np.random.choice(samples, size=len(samples), replace=True)
            bootstrapped.append(metric_fn(resampled))
        alpha = 1 - ci
        return (
            float(np.percentile(bootstrapped, 100 * alpha / 2)),
            float(np.percentile(bootstrapped, 100 * (1 - alpha / 2))),
        )

    def paired_ttest(self, ablation_samples: np.ndarray) -> tuple[float, float]:
        """
        Paired t-test: full model vs ablation variant, per-sample ECR.
        Returns (t_statistic, p_value).
        """
        from scipy import stats
        n = min(len(self.full_samples), len(ablation_samples))
        t_stat, p_val = stats.ttest_rel(
            self.full_samples[:n], ablation_samples[:n]
        )
        return float(t_stat), float(p_val)

    def report(self, ablation_samples: np.ndarray, variant_name: str):
        """Print significance test results for one ablation comparison."""
        t, p = self.paired_ttest(ablation_samples)
        ci_full   = self.bootstrap_ci(self.full_samples)
        ci_ablation = self.bootstrap_ci(ablation_samples)

        sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"\n  Full model vs {variant_name}")
        print(f"    Full model ECR:      {self.full_samples.mean():.4f} "
              f"[{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
        print(f"    Ablation ECR:        {ablation_samples.mean():.4f} "
              f"[{ci_ablation[0]:.4f}, {ci_ablation[1]:.4f}]")
        print(f"    t={t:.3f}, p={p:.4f} {sig_str}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpectraLM ablation runner")
    parser.add_argument("--data_dir",    type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="evals/results")
    parser.add_argument("--epochs",      type=int, default=60)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--quick",       action="store_true",
                        help="5 epochs per variant + 200 eval samples (for CI/testing)")
    parser.add_argument("--table_only",  action="store_true",
                        help="Skip training, load cached results and print table")
    args = parser.parse_args()

    runner = AblationRunner(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        quick=args.quick,
    )

    runner.run(skip_training=args.table_only)
    runner.print_table()
    runner.save_table()