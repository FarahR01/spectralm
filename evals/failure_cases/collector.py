"""
evals/failure_cases/collector.py

Harvests high-ECR predictions from evaluation runs and builds
a structured failure case dataset with automatic taxonomy tagging.

A failure case is any prediction where:
    ECR > threshold           (physics violation)
    OR Tanimoto < 0.3         (structurally wrong)
    OR valid_smiles = False   (generated invalid SMILES)

For each failure, the collector:
    1. Stores the observed spectrum and reconstructed spectrum
    2. Records the predicted vs. true SMILES
    3. Auto-tags the failure type (solvent, isomer, novel scaffold, overtone)
    4. Computes severity score for ranking
    5. Saves the annotated record to evals/failure_cases/annotated/

Usage:
    collector = FailureCaseCollector(model, device="cuda")
    cases = collector.collect_from_loader(test_loader, test_smiles, top_n=50)
    collector.save_all(cases, "evals/failure_cases/annotated")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectralm.models import SpectraLM
from spectralm.models.transformer import smiles_detokenise
from spectralm.physics.beer_lambert import BeerLambertConstraint
from spectralm.physics.group_frequencies import (
    GroupFrequencyChecker, WAVENUMBER_AXIS, GROUP_FREQUENCY_TABLE
)


# ══════════════════════════════════════════════════════════════════════════════
# Failure taxonomy
# ══════════════════════════════════════════════════════════════════════════════

FAILURE_TYPES = {
    "solvent_interference": (
        "The 2800–3000 cm⁻¹ region is masked by Nujol or solvent absorptions, "
        "causing the model to miss C–H stretches or misassign them."
    ),
    "regioisomer_confusion": (
        "Structural isomers (ortho/meta/para, or chain position) have nearly "
        "identical IR spectra. The model predicts the correct functional groups "
        "but assigns them to the wrong molecular position."
    ),
    "novel_scaffold": (
        "The molecular scaffold has fewer than 5 training examples. "
        "The model extrapolates SMILES fragments incorrectly."
    ),
    "overtone_band": (
        "The model misidentifies a harmonic overtone or combination band as a "
        "fundamental vibration, leading to incorrect functional group assignment."
    ),
    "invalid_smiles": (
        "The model generated a syntactically invalid SMILES string — "
        "typically due to unclosed rings or invalid valence."
    ),
    "low_confidence": (
        "The model's sequence log-probability is very low, indicating the "
        "prediction is uncertain. Often occurs with unusual molecular structures."
    ),
    "unknown": (
        "Failure cause not automatically determined. Requires manual inspection."
    ),
}

# Wavenumber regions for automatic failure typing
NUJOL_REGION       = (2850, 2960)
OVERTONE_REGION    = (1800, 2000)   # Combination/overtone bands for aromatics
FINGERPRINT_REGION = (500, 900)     # Often diagnostic for regioisomers


# ══════════════════════════════════════════════════════════════════════════════
# Data container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureCase:
    """Single annotated failure case."""
    case_id: str                    # e.g. "failure_0042"
    sample_idx: int

    # Spectra (stored as lists for JSON serialisation)
    observed_spectrum: list[float]
    reconstructed_spectrum: list[float]
    residual_spectrum: list[float]  # |observed - reconstructed|

    # Prediction
    predicted_smiles: str
    true_smiles: str
    valid_smiles: bool

    # Physics metrics
    ecr: float
    peak_position_error: float
    intensity_mse: float
    gf_recall: float
    gf_violations: list[str]        # violated group names

    # Chemistry metrics
    tanimoto: float
    mw_predicted: float
    mw_true: float
    formula_predicted: str
    formula_true: str

    # Taxonomy
    failure_type: str               # key from FAILURE_TYPES
    failure_description: str
    severity_score: float           # 0–1, higher = worse

    # Model diagnostics
    log_prob: float
    beam_rank: int                  # 0 = top beam prediction

    # Manual annotation fields (filled by human reviewer)
    manual_note: str = ""
    manually_reviewed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureCase":
        return cls(**d)

    @property
    def severity_label(self) -> str:
        if self.severity_score >= 0.7:  return "critical"
        if self.severity_score >= 0.4:  return "major"
        return "minor"


# ══════════════════════════════════════════════════════════════════════════════
# Auto-tagger
# ══════════════════════════════════════════════════════════════════════════════

class FailureTagger:
    """
    Automatically assigns a failure type to a prediction based on
    spectral features and prediction characteristics.

    Priority order (first match wins):
        1. invalid_smiles   — model generated unparseable SMILES
        2. solvent_interference — high residual in Nujol/solvent region
        3. overtone_band    — high residual in 1800–2000 cm⁻¹ region
        4. regioisomer_confusion — correct groups, wrong structure
        5. novel_scaffold   — very low log-probability
        6. low_confidence   — log-prob below threshold
        7. unknown
    """

    def __init__(self, gf_checker: GroupFrequencyChecker):
        self.checker = gf_checker

    def tag(
        self,
        observed: np.ndarray,
        reconstructed: np.ndarray,
        predicted_smiles: str,
        true_smiles: str,
        ecr: float,
        log_prob: float,
        tanimoto: float,
    ) -> tuple[str, float]:
        """
        Returns (failure_type, severity_score).
        severity_score is 0–1.
        """
        residual = np.abs(observed - reconstructed)
        wn = WAVENUMBER_AXIS

        # ── 1. Invalid SMILES ────────────────────────────────────────────
        try:
            from rdkit import Chem
            valid = Chem.MolFromSmiles(predicted_smiles) is not None
        except ImportError:
            valid = len(predicted_smiles) > 2
        if not valid:
            return "invalid_smiles", min(1.0, ecr * 2)

        # ── 2. Solvent interference ──────────────────────────────────────
        nujol_mask = (wn >= NUJOL_REGION[0]) & (wn <= NUJOL_REGION[1])
        nujol_residual = residual[nujol_mask].mean()
        if nujol_residual > 0.25:
            severity = min(1.0, nujol_residual * 2.5)
            return "solvent_interference", severity

        # ── 3. Overtone band misidentification ───────────────────────────
        ot_mask = (wn >= OVERTONE_REGION[0]) & (wn <= OVERTONE_REGION[1])
        ot_residual = residual[ot_mask].mean()
        # Overtone misidentification: residual is concentrated in quiet region
        overall_residual = residual.mean()
        if ot_residual > 0.15 and ot_residual > 2.0 * overall_residual:
            severity = min(1.0, ot_residual * 3.0)
            return "overtone_band", severity

        # ── 4. Regioisomer confusion ─────────────────────────────────────
        # High Tanimoto but wrong prediction → same groups, different topology
        if tanimoto > 0.5 and ecr > 0.15:
            # Both molecules have similar functional groups but ECR is still high
            # → peak positions are slightly off (regiochemistry)
            severity = min(1.0, ecr * 1.5 + (1 - tanimoto) * 0.5)
            return "regioisomer_confusion", severity

        # ── 5. Novel scaffold (low log-probability) ──────────────────────
        if log_prob < -8.0:
            severity = min(1.0, abs(log_prob) / 15.0 + ecr)
            return "novel_scaffold", severity

        # ── 6. Low confidence ────────────────────────────────────────────
        if log_prob < -4.0:
            severity = min(1.0, ecr + 0.2)
            return "low_confidence", severity

        # ── 7. Unknown ───────────────────────────────────────────────────
        return "unknown", min(1.0, ecr)


# ══════════════════════════════════════════════════════════════════════════════
# Collector
# ══════════════════════════════════════════════════════════════════════════════

class FailureCaseCollector:
    """
    Mines an evaluation run for the most instructive failure cases.

    Collects predictions where ECR > threshold or Tanimoto < 0.3,
    auto-tags each failure, computes severity, and returns a ranked list.
    """

    def __init__(
        self,
        model: SpectraLM,
        device: str | torch.device = "cpu",
        ecr_threshold: float = 0.15,     # lower than implausibility — catch borderline cases
        tanimoto_threshold: float = 0.40,
        beam_size: int = 1,
    ):
        self.model     = model
        self.device    = torch.device(device)
        self.ecr_threshold       = ecr_threshold
        self.tanimoto_threshold  = tanimoto_threshold
        self.beam_size           = beam_size

        self.bl_constraint = BeerLambertConstraint()
        self.gf_checker    = GroupFrequencyChecker()
        self.tagger        = FailureTagger(self.gf_checker)

        self.model.eval()
        self.model.to(self.device)

    def collect_from_loader(
        self,
        test_loader,
        test_smiles: list[str],
        top_n: int = 50,
    ) -> list[FailureCase]:
        """
        Run inference on test_loader, collect top_n worst failures.

        Returns list of FailureCase objects sorted by severity_score descending.
        """
        all_cases: list[FailureCase] = []
        sample_idx = 0

        for batch in tqdm(test_loader, desc="Collecting failures"):
            spectra    = batch["spectrum"].to(self.device)
            B          = spectra.size(0)

            with torch.no_grad():
                results = self.model.predict(spectra, beam_size=self.beam_size)

            for i in range(B):
                true_smiles = (test_smiles[sample_idx]
                               if sample_idx < len(test_smiles) else "")
                pred_tokens  = results["smiles_tokens"][i].cpu().tolist()
                pred_smiles  = smiles_detokenise(pred_tokens)
                ecr_val      = float(results["ecr"][i].cpu())
                log_prob_val = float(results["log_probs"][i].cpu())

                # Compute Tanimoto
                tanimoto = self._tanimoto(pred_smiles, true_smiles)

                # Is this a failure?
                is_failure = (
                    ecr_val > self.ecr_threshold
                    or tanimoto < self.tanimoto_threshold
                    or not self._valid_smiles(pred_smiles)
                )

                if is_failure:
                    case = self._build_case(
                        case_id=f"failure_{sample_idx:04d}",
                        sample_idx=sample_idx,
                        observed=spectra[i].cpu().numpy(),
                        results=results,
                        i=i,
                        pred_smiles=pred_smiles,
                        true_smiles=true_smiles,
                        ecr_val=ecr_val,
                        log_prob_val=log_prob_val,
                        tanimoto=tanimoto,
                    )
                    all_cases.append(case)

                sample_idx += 1

        # Sort by severity descending, return top_n
        all_cases.sort(key=lambda c: c.severity_score, reverse=True)
        selected = all_cases[:top_n]

        print(f"\nCollected {len(all_cases)} failures from "
              f"{sample_idx} predictions")
        print(f"Returning top {len(selected)} by severity")
        self._print_taxonomy_summary(selected)

        return selected

    def collect_from_eval_report(
        self,
        report,     # EvalReport from domain_residuals.py
        test_loader,
        top_n: int = 50,
    ) -> list[FailureCase]:
        """
        Alternative: build failure cases from a pre-computed EvalReport.
        Avoids re-running inference.
        """
        from evals.domain_residuals import SampleResult

        # Find worst samples
        worst = sorted(
            report.sample_results,
            key=lambda s: s.ecr + (1.0 - max(s.tanimoto, 0)),
            reverse=True,
        )[:top_n]

        cases: list[FailureCase] = []
        for sr in worst:
            # Use stored spectra from evaluation results
            obs_np   = sr.observed_spectrum if sr.observed_spectrum.size > 0 else np.zeros(1800)
            recon_np = sr.reconstructed_spectrum if sr.reconstructed_spectrum.size > 0 else np.zeros(1800)

            failure_type, severity = self.tagger.tag(
                observed=obs_np,
                reconstructed=recon_np,
                predicted_smiles=sr.predicted_smiles,
                true_smiles=sr.true_smiles,
                ecr=sr.ecr,
                log_prob=sr.log_prob,
                tanimoto=sr.tanimoto,
            )

            case = FailureCase(
                case_id=f"failure_{sr.sample_idx:04d}",
                sample_idx=sr.sample_idx,
                observed_spectrum=obs_np.tolist(),
                reconstructed_spectrum=recon_np.tolist(),
                residual_spectrum=np.abs(obs_np - recon_np).tolist(),
                predicted_smiles=sr.predicted_smiles,
                true_smiles=sr.true_smiles,
                valid_smiles=sr.valid_smiles,
                ecr=sr.ecr,
                peak_position_error=sr.peak_position_error,
                intensity_mse=sr.intensity_mse,
                gf_recall=sr.gf_recall,
                gf_violations=[],
                tanimoto=sr.tanimoto,
                mw_predicted=-1.0,
                mw_true=-1.0,
                formula_predicted="",
                formula_true="",
                failure_type=failure_type,
                failure_description=FAILURE_TYPES[failure_type],
                severity_score=severity,
                log_prob=sr.log_prob,
                beam_rank=0,
            )
            cases.append(case)

        return cases

    def save_all(
        self,
        cases: list[FailureCase],
        output_dir: str | Path,
    ):
        """
        Save all failure cases to JSON files.
        One file per case + a summary index.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        index = []
        for case in cases:
            case_path = output_dir / f"{case.case_id}.json"
            with open(case_path, "w") as f:
                json.dump(case.to_dict(), f, indent=2)

            index.append({
                "case_id":       case.case_id,
                "failure_type":  case.failure_type,
                "severity":      case.severity_label,
                "severity_score": round(case.severity_score, 4),
                "ecr":           round(case.ecr, 4),
                "tanimoto":      round(case.tanimoto, 4),
                "predicted_smiles": case.predicted_smiles,
                "true_smiles":   case.true_smiles,
            })

        with open(output_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        print(f"\n✓ Saved {len(cases)} failure cases → {output_dir}")
        print(f"  Index → {output_dir / 'index.json'}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_case(
        self,
        case_id: str,
        sample_idx: int,
        observed: np.ndarray,
        results: dict,
        i: int,
        pred_smiles: str,
        true_smiles: str,
        ecr_val: float,
        log_prob_val: float,
        tanimoto: float,
    ) -> FailureCase:
        """Construct a FailureCase from raw inference outputs."""
        recon_np = results["reconstructed_spec"][i].cpu().numpy()
        residual = np.abs(observed - recon_np)

        # Group frequency violations
        gf_report  = self.gf_checker.check(observed, pred_smiles)
        gf_viols   = [v.group_name for v in gf_report.violations]
        gf_recall  = gf_report.recall

        # Per-sample BL report
        spec_t  = torch.from_numpy(observed).unsqueeze(0)
        recon_t = results["reconstructed_spec"][i:i+1].cpu()
        _, bl   = self.bl_constraint(spec_t, recon_t)
        peak_err = float(bl.peak_position_error[0])
        int_mse  = float(bl.intensity_mse[0])

        # MW and formula
        mw_pred, mw_true, form_pred, form_true = self._chem_metrics(
            pred_smiles, true_smiles
        )

        # Auto-tag
        failure_type, severity = self.tagger.tag(
            observed=observed,
            reconstructed=recon_np,
            predicted_smiles=pred_smiles,
            true_smiles=true_smiles,
            ecr=ecr_val,
            log_prob=log_prob_val,
            tanimoto=tanimoto,
        )

        return FailureCase(
            case_id=case_id,
            sample_idx=sample_idx,
            observed_spectrum=observed.tolist(),
            reconstructed_spectrum=recon_np.tolist(),
            residual_spectrum=residual.tolist(),
            predicted_smiles=pred_smiles,
            true_smiles=true_smiles,
            valid_smiles=self._valid_smiles(pred_smiles),
            ecr=ecr_val,
            peak_position_error=peak_err,
            intensity_mse=int_mse,
            gf_recall=gf_recall,
            gf_violations=gf_viols,
            tanimoto=tanimoto,
            mw_predicted=mw_pred,
            mw_true=mw_true,
            formula_predicted=form_pred,
            formula_true=form_true,
            failure_type=failure_type,
            failure_description=FAILURE_TYPES[failure_type],
            severity_score=severity,
            log_prob=log_prob_val,
            beam_rank=0,
        )

    def _valid_smiles(self, smiles: str) -> bool:
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles) is not None
        except ImportError:
            return len(smiles) > 1

    def _tanimoto(self, smiles_a: str, smiles_b: str) -> float:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator, DataStructs
            mol_a = Chem.MolFromSmiles(smiles_a)
            mol_b = Chem.MolFromSmiles(smiles_b)
            if mol_a is None or mol_b is None:
                return 0.0
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            return DataStructs.TanimotoSimilarity(
                gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b)
            )
        except Exception:
            return 0.0

    def _chem_metrics(
        self, pred_smiles: str, true_smiles: str
    ) -> tuple[float, float, str, str]:
        """Returns (mw_pred, mw_true, formula_pred, formula_true)."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            mol_p = Chem.MolFromSmiles(pred_smiles)
            mol_t = Chem.MolFromSmiles(true_smiles)
            mw_p  = Descriptors.MolWt(mol_p) if mol_p else -1.0
            mw_t  = Descriptors.MolWt(mol_t) if mol_t else -1.0
            fp    = rdMolDescriptors.CalcMolFormula(mol_p) if mol_p else ""
            ft    = rdMolDescriptors.CalcMolFormula(mol_t) if mol_t else ""
            return mw_p, mw_t, fp, ft
        except Exception:
            return -1.0, -1.0, "", ""

    def _print_taxonomy_summary(self, cases: list[FailureCase]):
        """Print failure type distribution."""
        from collections import Counter
        counts = Counter(c.failure_type for c in cases)
        print("\nFailure taxonomy:")
        for ftype, count in counts.most_common():
            pct = count / len(cases)
            bar = "█" * int(pct * 30)
            print(f"  {ftype:<28} {bar:<32} {count:3d}  ({pct:.1%})")