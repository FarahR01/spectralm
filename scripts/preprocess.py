"""
scripts/preprocess.py

Converts raw .jdx files (data/raw/) into model-ready tensors (data/processed/).
Also generates the train/val/test splits (data/splits/).

Pipeline:
    data/raw/**/*.jdx
        ↓  parse_jdx()        — JCAMP-DX → numpy array
        ↓  quality_filter()   — SNR check, label consistency audit
        ↓  normalise()        — interpolate to 1800-pt axis, scale to [0,1]
        ↓  stratified_split() — by MW bracket, random seed 42
        ↓
    data/processed/
        ├── train_spectra.pt     (N_train, 1800) float32
        ├── train_smiles.json    [N_train SMILES strings]
        ├── val_spectra.pt
        ├── val_smiles.json
        ├── test_spectra.pt
        ├── test_smiles.json
        └── dataset_stats.json   metadata for reproducibility

    data/splits/
        ├── train_indices.json
        ├── val_indices.json
        ├── test_indices.json
        └── split_manifest.json  CAS numbers + split assignment

Usage:
    python scripts/preprocess.py --raw_dir data/raw --out_dir data/processed
    python scripts/preprocess.py --raw_dir data/raw --out_dir data/processed --min_snr 8
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import jcamp
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectralm.physics.group_frequencies import (
    WAVENUMBER_AXIS, GroupFrequencyChecker, GROUP_FREQUENCY_TABLE
)

console = Console()
checker = GroupFrequencyChecker()


# ══════════════════════════════════════════════════════════════════════════════
# Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_jdx(path: Path) -> dict | None:
    """
    Parse a JCAMP-DX .jdx file into a normalised spectrum dict.

    Returns None if the file is unparseable, has insufficient data,
    or falls entirely outside the target wavenumber range.
    """
    try:
        data = jcamp.jcamp_readfile(str(path))
    except Exception:
        return None

    x = np.asarray(data.get("x", []), dtype=float)
    y = np.asarray(data.get("y", []), dtype=float)

    if len(x) < 50 or len(y) < 50:
        return None

    # ── Ensure ascending wavenumber ───────────────────────────────────────
    if x[0] > x[-1]:
        x, y = x[::-1].copy(), y[::-1].copy()

    # ── Transmittance → Absorbance ────────────────────────────────────────
    yunits = str(data.get("yunits", "")).lower()
    if any(k in yunits for k in ("transmittance", "transmission", "%t")):
        y = np.clip(y / 100.0 if y.max() > 5 else y, 1e-6, 1.0)
        y = -np.log10(y)
    elif "reflectance" in yunits:
        y = np.clip(y / 100.0 if y.max() > 5 else y, 1e-6, 1.0)
        y = -np.log10(1.0 - y + 1e-6)

    # ── Clip to target range ──────────────────────────────────────────────
    mask = (x >= 380.0) & (x <= 4050.0)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return None

    # ── Remove negative absorbance (instrument artefact) ─────────────────
    y = np.clip(y, 0.0, None)

    # ── Interpolate onto standard 1800-point axis ─────────────────────────
    interp = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    y_std = interp(WAVENUMBER_AXIS).astype(np.float32)

    # ── Normalise to [0, 1] ───────────────────────────────────────────────
    y_max = y_std.max()
    if y_max < 1e-6:
        return None
    y_norm = y_std / y_max

    # ── Extract metadata ──────────────────────────────────────────────────
    smiles = (
        data.get("$smiles") or
        data.get("smiles") or
        data.get("##$smiles") or
        None
    )
    if smiles:
        smiles = str(smiles).strip()

    return {
        "spectrum":  y_norm,
        "smiles":    smiles,
        "cas":       str(data.get("cas registry no", path.stem)).strip(),
        "title":     str(data.get("title", "Unknown")).strip(),
        "path":      str(path),
        "x_range":   (float(x.min()), float(x.max())),
        "n_points":  int(len(x)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Quality filtering
# ══════════════════════════════════════════════════════════════════════════════

def estimate_snr(spectrum: np.ndarray) -> float:
    """
    SNR estimate using the 1800–2000 cm⁻¹ quiet region as noise floor.
    Most organic molecules have no fundamental absorptions here.
    """
    quiet = (WAVENUMBER_AXIS >= 1800) & (WAVENUMBER_AXIS <= 2000)
    noise = spectrum[quiet].std()
    if noise < 1e-8:
        return 999.0
    return float(spectrum.max() / noise)


def validate_smiles(smiles: str) -> bool:
    """Check SMILES is parseable by RDKit."""
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return len(smiles) > 2  # fallback


def label_consistency_score(spectrum: np.ndarray, smiles: str) -> float:
    """
    Group frequency recall score. Values < 0.5 indicate likely label errors.
    These records are kept but flagged for down-weighting.
    """
    try:
        report = checker.check(spectrum, smiles)
        return report.recall if report.total_groups_checked > 0 else 1.0
    except Exception:
        return 1.0


def quality_filter(
    record: dict,
    min_snr: float = 10.0,
    min_label_consistency: float = 0.0,  # 0 = keep all, just flag
) -> tuple[bool, dict]:
    """
    Filter a parsed record for quality.

    Returns:
        (passes, quality_dict)
        passes       : whether to include this record
        quality_dict : diagnostic metrics attached to the record
    """
    quality = {}

    # ── SNR check ─────────────────────────────────────────────────────────
    snr = estimate_snr(record["spectrum"])
    quality["snr"] = round(snr, 2)
    if snr < min_snr:
        return False, quality

    # ── SMILES check ──────────────────────────────────────────────────────
    if not record.get("smiles"):
        return False, quality
    if not validate_smiles(record["smiles"]):
        return False, quality

    # ── Label consistency (flag but don't filter unless very low) ─────────
    consistency = label_consistency_score(record["spectrum"], record["smiles"])
    quality["label_consistency"] = round(consistency, 3)
    quality["label_suspect"] = consistency < 0.5

    if min_label_consistency > 0 and consistency < min_label_consistency:
        return False, quality

    quality["passes"] = True
    return True, quality


# ══════════════════════════════════════════════════════════════════════════════
# Molecular weight bracket for stratified splitting
# ══════════════════════════════════════════════════════════════════════════════

def get_mw_stratum(smiles: str) -> int:
    """Return stratum index (0–4) for stratified MW splitting."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 2
        mw = Descriptors.MolWt(mol)
        if mw < 100:   return 0
        if mw < 200:   return 1
        if mw < 300:   return 2
        if mw < 500:   return 3
        return 4
    except Exception:
        return 2


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    splits_dir: Path,
    min_snr: float = 10.0,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    random_seed: int = 42,
    smooth_sigma: float = 0.0,  # 0 = no smoothing
) -> dict:
    """
    Full preprocessing pipeline. Returns dataset statistics dict.
    """
    raw_dir   = Path(raw_dir)
    out_dir   = Path(out_dir)
    splits_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # ── 1. Discover all .jdx files ─────────────────────────────────────────
    jdx_files = sorted(raw_dir.glob("**/*.jdx"))
    console.print(f"\n[bold]Found {len(jdx_files):,} .jdx files[/bold] in {raw_dir}")

    # ── 2. Parse ───────────────────────────────────────────────────────────
    console.print("Parsing .jdx files...")
    raw_records, parse_failures = [], 0
    for f in tqdm(jdx_files, desc="Parsing", unit="file"):
        r = parse_jdx(f)
        if r is not None:
            raw_records.append(r)
        else:
            parse_failures += 1

    console.print(f"  Parsed:  [green]{len(raw_records):,}[/green]")
    console.print(f"  Failed:  [red]{parse_failures:,}[/red]"
                  f"  ({parse_failures / max(len(jdx_files), 1):.1%})")

    # ── 3. Quality filter ──────────────────────────────────────────────────
    console.print(f"Quality filtering (min_snr={min_snr})...")
    good_records, filter_reasons = [], []
    n_low_snr = n_no_smiles = n_bad_smiles = n_suspect = 0

    for r in tqdm(raw_records, desc="Filtering", unit="record"):
        passes, quality = quality_filter(r, min_snr=min_snr)
        r["quality"] = quality
        if passes:
            # Optional smoothing
            if smooth_sigma > 0:
                r["spectrum"] = gaussian_filter1d(r["spectrum"], sigma=smooth_sigma)
            good_records.append(r)
            if quality.get("label_suspect"):
                n_suspect += 1
        else:
            if quality.get("snr", 999) < min_snr:
                n_low_snr += 1
            elif not r.get("smiles"):
                n_no_smiles += 1
            else:
                n_bad_smiles += 1

    console.print(f"  Kept:           [green]{len(good_records):,}[/green]")
    console.print(f"  Low SNR:        [yellow]{n_low_snr:,}[/yellow]")
    console.print(f"  Missing SMILES: [yellow]{n_no_smiles:,}[/yellow]")
    console.print(f"  Invalid SMILES: [yellow]{n_bad_smiles:,}[/yellow]")
    console.print(f"  Suspect labels: [yellow]{n_suspect:,}[/yellow]"
                  f"  (kept, flagged for down-weighting)")

    if len(good_records) < 50:
        console.print("[bold red]Too few records after filtering. "
                      "Check raw data or lower --min_snr.[/bold red]")
        return {}

    # ── 4. Stratified train/val/test split ─────────────────────────────────
    console.print("Splitting dataset...")
    strata = [get_mw_stratum(r["smiles"]) for r in good_records]
    all_idx = list(range(len(good_records)))

    # Check stratum sizes and merge undersized strata
    from collections import Counter
    strata_counts = Counter(strata)
    undersized = [s for s, count in strata_counts.items() if count < 2]
    
    if undersized:
        console.print(f"  [yellow]Merging undersized strata: {undersized}[/yellow]")
        # Merge undersized strata into nearest sized stratum
        merge_into = max(strata_counts.keys())
        strata = [merge_into if s in undersized else s for s in strata]
        strata_counts = Counter(strata)
    
    strata_counts_str = ", ".join(f"{s}: {c}" for s, c in sorted(strata_counts.items()))
    console.print(f"  Stratum distribution: {strata_counts_str}")

    # Split: train + temp
    test_val_frac = val_frac + test_frac
    train_idx, temp_idx, _, temp_strata = train_test_split(
        all_idx, strata,
        test_size=test_val_frac,
        stratify=strata,
        random_state=random_seed,
    )
# Split temp → val + test
    # Drop stratify on the second split — temp set is too small
    # for all strata to have ≥2 members
    relative_test = test_frac / test_val_frac
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=None,
        random_state=random_seed,
    )

    console.print(f"  Train: [green]{len(train_idx):,}[/green]"
                  f"  Val: [cyan]{len(val_idx):,}[/cyan]"
                  f"  Test: [yellow]{len(test_idx):,}[/yellow]")

    # ── 5. Save tensors and SMILES lists ───────────────────────────────────
    console.print("Saving processed files...")
    split_manifest = []

    for split_name, idxs in [
        ("train", train_idx),
        ("val",   val_idx),
        ("test",  test_idx),
    ]:
        spectra_list = [good_records[i]["spectrum"] for i in idxs]
        smiles_list  = [good_records[i]["smiles"]   for i in idxs]

        spectra_tensor = torch.tensor(np.stack(spectra_list), dtype=torch.float32)
        torch.save(spectra_tensor, out_dir / f"{split_name}_spectra.pt")

        with open(out_dir / f"{split_name}_smiles.json", "w") as f:
            json.dump(smiles_list, f, indent=2)

        # Split manifest entry
        for i in idxs:
            r = good_records[i]
            split_manifest.append({
                "cas":              r["cas"],
                "title":            r["title"],
                "smiles":           r["smiles"],
                "split":            split_name,
                "snr":              r["quality"].get("snr"),
                "label_consistency": r["quality"].get("label_consistency"),
                "label_suspect":    r["quality"].get("label_suspect", False),
            })

        console.print(f"  [green]✓[/green] {split_name}: "
                      f"{len(idxs):,} samples saved")

    # ── 6. Save split indices ──────────────────────────────────────────────
    for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(splits_dir / f"{split_name}_indices.json", "w") as f:
            json.dump(idxs, f)

    with open(splits_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    # ── 7. Compute and save dataset statistics ─────────────────────────────
    all_spectra = np.stack([r["spectrum"] for r in good_records])
    snr_values  = [r["quality"]["snr"] for r in good_records]

    # Functional group distribution
    group_counts: dict[str, int] = {}
    for gf in GROUP_FREQUENCY_TABLE:
        count = sum(
            1 for r in good_records
            if label_consistency_score(r["spectrum"], r["smiles"]) > 0
        )
        group_counts[gf.name] = count

    stats = {
        "total_compounds":    len(good_records),
        "train_size":         len(train_idx),
        "val_size":           len(val_idx),
        "test_size":          len(test_idx),
        "parse_failures":     parse_failures,
        "filtered_low_snr":   n_low_snr,
        "filtered_no_smiles": n_no_smiles,
        "suspect_labels":     n_suspect,
        "snr_mean":           round(float(np.mean(snr_values)), 2),
        "snr_median":         round(float(np.median(snr_values)), 2),
        "snr_min":            round(float(np.min(snr_values)), 2),
        "spectrum_mean":      round(float(all_spectra.mean()), 4),
        "spectrum_std":       round(float(all_spectra.std()), 4),
        "wavenumber_axis": {
            "min":    float(WAVENUMBER_AXIS[0]),
            "max":    float(WAVENUMBER_AXIS[-1]),
            "steps":  int(len(WAVENUMBER_AXIS)),
            "resolution_cm": round(float(WAVENUMBER_AXIS[1] - WAVENUMBER_AXIS[0]), 3),
        },
        "random_seed":   random_seed,
        "min_snr_filter": min_snr,
        "preprocessing_version": "1.0.0",
    }

    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 8. Print summary ───────────────────────────────────────────────────
    table = Table(title="Dataset Statistics", show_header=True)
    table.add_column("Metric",  style="bold", width=30)
    table.add_column("Value",   justify="right")
    for k, v in stats.items():
        if isinstance(v, dict):
            continue
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    console.print(f"\n[bold green]✓ Preprocessing complete.[/bold green]")
    console.print(f"  Processed tensors → {out_dir}")
    console.print(f"  Split manifests   → {splits_dir}")
    console.print(f"\n  Next: [bold]python scripts/train.py --data_dir {out_dir}[/bold]")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NIST IR spectra")
    parser.add_argument("--raw_dir",    type=str, default="data/raw",
                        help="Input directory with .jdx files")
    parser.add_argument("--out_dir",    type=str, default="data/processed",
                        help="Output directory for tensors + SMILES")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Output directory for index files")
    parser.add_argument("--min_snr",    type=float, default=10.0,
                        help="Minimum SNR to include a spectrum (default: 10.0)")
    parser.add_argument("--val_frac",   type=float, default=0.10)
    parser.add_argument("--test_frac",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--smooth",     type=float, default=0.0,
                        help="Gaussian smoothing sigma (0 = disabled)")
    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        splits_dir=Path(args.splits_dir),
        min_snr=args.min_snr,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_seed=args.seed,
        smooth_sigma=args.smooth,
    )
