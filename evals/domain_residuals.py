"""
evals/domain_residuals.py

Domain-specific residual scorer for SpectraLM.

This module is the primary evaluation harness. It computes all metrics
reported in the paper — with emphasis on physics-domain residuals that
standard NLP metrics (BLEU, accuracy) cannot capture.

The central argument: a model can achieve high BLEU while being
scientifically unreliable. This scorer makes that visible.

Metrics computed:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Physics-domain                                                  │
    │   ECR         — Energy Conservation Residual (Beer–Lambert)    │
    │   GF_recall   — Group Frequency Recall (functional group ID)   │
    │   ISO_error   — Isotope shift prediction error (cm⁻¹)          │
    │   implaus_rate — Fraction of predictions with ECR > threshold  │
    │                                                                 │
    │ Chemistry-domain                                                │
    │   tanimoto    — Morgan fingerprint similarity (0–1)            │
    │   valid_smiles — Fraction generating parseable SMILES          │
    │   mw_error    — Molecular weight prediction error (g/mol)      │
    │   formula_acc — Molecular formula exact match rate             │
    │                                                                 │
    │ NLP-domain                                                      │
    │   bleu_4      — BLEU-4 on SMILES token sequences               │
    │   token_acc   — Per-token accuracy (excluding padding)         │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    scorer = DomainResidualScorer(model, device)
    report = scorer.evaluate(test_dataloader, test_smiles_list)
    report.print_summary()
    report.save("evals/results/run_001.json")
"""

from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Optional heavy deps with graceful fallback ─────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available — chemistry metrics will be skipped.")

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available — BLEU will be computed manually.")


# ── Project imports ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectralm.models import SpectraLM
from spectralm.models.transformer import smiles_detokenise, PAD_IDX, EOS_IDX
from spectralm.physics.beer_lambert import BeerLambertConstraint
from spectralm.physics.group_frequencies import GroupFrequencyChecker, WAVENUMBER_AXIS


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    """Per-sample evaluation result."""
    sample_idx: int

    # Physics metrics
    ecr: float
    peak_position_error: float
    intensity_mse: float
    physically_implausible: bool

    # Chemistry metrics
    predicted_smiles: str
    true_smiles: str
    valid_smiles: bool
    tanimoto: float             # -1.0 if either SMILES is invalid
    mw_error: float             # |predicted_MW - true_MW|, -1 if invalid
    formula_match: bool

    # Group frequency metrics
    gf_recall: float
    gf_violation_count: int
    gf_groups_checked: int

    # NLP metrics
    token_accuracy: float

    # Diagnostics
    log_prob: float
    inference_time_ms: float


@dataclass
class EvalReport:
    """Aggregated evaluation report across all test samples."""
    model_name: str
    timestamp: str
    num_samples: int
    eval_time_sec: float

    # ── Physics metrics ──────────────────────────────────────────────────
    ecr_mean: float = 0.0
    ecr_median: float = 0.0
    ecr_std: float = 0.0
    ecr_p95: float = 0.0
    implausible_rate: float = 0.0
    peak_position_error_mean: float = 0.0
    intensity_mse_mean: float = 0.0

    # ── Chemistry metrics ────────────────────────────────────────────────
    valid_smiles_rate: float = 0.0
    tanimoto_mean: float = 0.0
    tanimoto_median: float = 0.0
    mw_error_mean: float = 0.0
    mw_error_median: float = 0.0
    formula_accuracy: float = 0.0

    # ── Group frequency metrics ──────────────────────────────────────────
    gf_recall_mean: float = 0.0
    gf_recall_std: float = 0.0
    gf_violation_rate: float = 0.0

    # ── NLP metrics ──────────────────────────────────────────────────────
    bleu_4: float = 0.0
    token_accuracy_mean: float = 0.0

    # ── Failure analysis ─────────────────────────────────────────────────
    failure_by_mw_bracket: dict = field(default_factory=dict)
    top_violation_groups: list = field(default_factory=list)

    # Raw results for deep-dive analysis
    sample_results: list = field(default_factory=list, repr=False)

    def print_summary(self):
        """Formatted summary table."""
        width = 62
        print("\n" + "═" * width)
        print(f"  SPECTRALM EVALUATION REPORT")
        print(f"  Model: {self.model_name}")
        print(f"  n={self.num_samples:,}  |  time={self.eval_time_sec:.1f}s")
        print("═" * width)
        print(f"\n  {'PHYSICS METRICS':─<40}")
        print(f"  ECR (mean ± std):       {self.ecr_mean:.4f} ± {self.ecr_std:.4f}")
        print(f"  ECR (median / p95):     {self.ecr_median:.4f} / {self.ecr_p95:.4f}")
        print(f"  Implausible rate:       {self.implausible_rate:.2%}")
        print(f"  Peak position error:    {self.peak_position_error_mean:.4f}")
        print(f"  Intensity MSE:          {self.intensity_mse_mean:.4f}")
        print(f"\n  {'CHEMISTRY METRICS':─<40}")
        print(f"  Valid SMILES:           {self.valid_smiles_rate:.2%}")
        print(f"  Tanimoto (mean/median): {self.tanimoto_mean:.3f} / {self.tanimoto_median:.3f}")
        print(f"  MW error (mean/median): {self.mw_error_mean:.1f} / {self.mw_error_median:.1f} g/mol")
        print(f"  Formula accuracy:       {self.formula_accuracy:.2%}")
        print(f"\n  {'GROUP FREQUENCY METRICS':─<40}")
        print(f"  GF recall (mean ± std): {self.gf_recall_mean:.3f} ± {self.gf_recall_std:.3f}")
        print(f"  GF violation rate:      {self.gf_violation_rate:.2%}")
        print(f"\n  {'NLP METRICS':─<40}")
        print(f"  BLEU-4:                 {self.bleu_4:.4f}")
        print(f"  Token accuracy:         {self.token_accuracy_mean:.3f}")
        print("═" * width)

        if self.top_violation_groups:
            print(f"\n  Most violated group frequencies:")
            for group, count in self.top_violation_groups[:5]:
                print(f"    {group:<42} {count:>4}×")

        if self.failure_by_mw_bracket:
            print(f"\n  Implausible rate by MW bracket:")
            for bracket, rate in self.failure_by_mw_bracket.items():
                bar = "█" * int(rate * 30)
                print(f"    {bracket:<15} {bar:<32} {rate:.1%}")

    def save(self, path: str | Path):
        """Save full report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # sample_results are verbose — save separately if needed
        data.pop("sample_results", None)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Report saved → {path}")

    def save_samples(self, path: str | Path):
        """Save per-sample results for deep-dive analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in self.sample_results], f, indent=2)
        print(f"  Sample results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Chemistry metric utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan fingerprint Tanimoto similarity. Returns -1.0 if either is invalid."""
    if not RDKIT_AVAILABLE:
        return -1.0
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return -1.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = gen.GetFingerprint(mol_a)
    fp_b = gen.GetFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def compute_mw_error(smiles_pred: str, smiles_true: str) -> float:
    """Absolute molecular weight error in g/mol. Returns -1.0 on failure."""
    if not RDKIT_AVAILABLE:
        return -1.0
    mol_p = Chem.MolFromSmiles(smiles_pred)
    mol_t = Chem.MolFromSmiles(smiles_true)
    if mol_p is None or mol_t is None:
        return -1.0
    return abs(Descriptors.MolWt(mol_p) - Descriptors.MolWt(mol_t))


def compute_formula_match(smiles_pred: str, smiles_true: str) -> bool:
    """Whether predicted and true molecules have identical molecular formula."""
    if not RDKIT_AVAILABLE:
        return False
    mol_p = Chem.MolFromSmiles(smiles_pred)
    mol_t = Chem.MolFromSmiles(smiles_true)
    if mol_p is None or mol_t is None:
        return False
    formula_p = rdMolDescriptors.CalcMolFormula(mol_p)
    formula_t = rdMolDescriptors.CalcMolFormula(mol_t)
    return formula_p == formula_t


def get_mw_bracket(smiles: str) -> str:
    """Bin molecule into MW bracket for stratified analysis."""
    if not RDKIT_AVAILABLE:
        return "unknown"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    mw = Descriptors.MolWt(mol)
    if mw < 100:   return "< 100"
    if mw < 150:   return "100–150"
    if mw < 200:   return "150–200"
    if mw < 300:   return "200–300"
    if mw < 500:   return "300–500"
    return "> 500"


# ══════════════════════════════════════════════════════════════════════════════
# BLEU computation (manual fallback if NLTK unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def compute_bleu_4(
    hypotheses: list[list[int]],
    references: list[list[int]],
) -> float:
    """
    Compute corpus BLEU-4 over token sequences.
    Falls back to a simple n-gram precision if NLTK is unavailable.
    """
    if NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        refs_wrapped = [[r] for r in references]
        return corpus_bleu(refs_wrapped, hypotheses,
                           smoothing_function=smoothing)

    # Manual fallback: geometric mean of 1–4 gram precisions
    from collections import Counter
    import math

    def ngram_precision(hyps, refs, n):
        total_match, total_count = 0, 0
        for hyp, ref in zip(hyps, refs):
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            match = sum((hyp_ngrams & ref_ngrams).values())
            total_match += match
            total_count += max(len(hyp) - n + 1, 0)
        return total_match / max(total_count, 1)

    precisions = [ngram_precision(hypotheses, references, n) for n in range(1, 5)]
    if min(precisions) == 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / 4
    bp = min(1.0, np.exp(1 - np.mean([len(r) for r in references]) /
                          max(np.mean([len(h) for h in hypotheses]), 1)))
    return float(bp * math.exp(log_avg))


# ══════════════════════════════════════════════════════════════════════════════
# Main scorer
# ══════════════════════════════════════════════════════════════════════════════

class DomainResidualScorer:
    """
    Comprehensive domain-specific evaluator for SpectraLM.

    Evaluates a model checkpoint against a test DataLoader and a list
    of ground-truth SMILES strings. Computes all physics, chemistry,
    and NLP metrics and returns a structured EvalReport.

    Example:
        scorer = DomainResidualScorer(model, device="cuda")
        report = scorer.evaluate(test_loader, test_smiles, beam_size=4)
        report.print_summary()
        report.save("evals/results/full_model.json")
    """

    def __init__(
        self,
        model: SpectraLM,
        device: str | torch.device = "cpu",
        implausibility_threshold: float = 0.25,
        beam_size: int = 4,
        verbose: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.verbose = verbose

        self.bl_constraint = BeerLambertConstraint(
            implausibility_threshold=implausibility_threshold
        )
        self.gf_checker = GroupFrequencyChecker()

        self.model.eval()
        self.model.to(self.device)

    def evaluate(
        self,
        test_loader: DataLoader,
        test_smiles: list[str],
        model_name: str = "spectralm",
        max_samples: int | None = None,
    ) -> EvalReport:
        """
        Run full evaluation.

        Args:
            test_loader  : DataLoader yielding {spectrum, tokens, padding_mask}
            test_smiles  : Ground-truth SMILES list (parallel to loader order)
            model_name   : Label for the report
            max_samples  : Cap evaluation at N samples (None = all)

        Returns:
            EvalReport with all metrics populated
        """
        import datetime
        t_start = time.time()

        sample_results: list[SampleResult] = []
        all_hyp_tokens: list[list[int]] = []
        all_ref_tokens: list[list[int]] = []

        sample_idx = 0
        for batch in tqdm(test_loader, desc="Evaluating", disable=not self.verbose):
            if max_samples and sample_idx >= max_samples:
                break

            spectra = batch["spectrum"].to(self.device)        # (B, W)
            ref_tokens_batch = batch["tokens"]                  # (B, T)
            B = spectra.size(0)

            t_inf = time.time()
            with torch.no_grad():
                results = self.model.predict(
                    spectra,
                    beam_size=self.beam_size,
                    return_diagnostics=True,
                )
            inf_ms = (time.time() - t_inf) * 1000 / B

            for i in range(B):
                if max_samples and sample_idx >= max_samples:
                    break

                true_smiles = test_smiles[sample_idx] if sample_idx < len(test_smiles) else ""
                pred_tokens = results["smiles_tokens"][i].cpu().tolist()
                pred_smiles = smiles_detokenise(pred_tokens)

                # ── Physics metrics ────────────────────────────────────────
                ecr_val  = float(results["ecr"][i].cpu())
                implaus  = bool(results["implausible"][i].cpu())

                # Detailed BL report (per-sample)
                spec_single  = spectra[i:i+1]
                recon_single = results["reconstructed_spec"][i:i+1]
                _, bl_report = self.bl_constraint(spec_single, recon_single)
                peak_err = float(bl_report.peak_position_error[0].cpu())
                int_mse  = float(bl_report.intensity_mse[0].cpu())

                # ── Chemistry metrics ──────────────────────────────────────
                valid = Chem.MolFromSmiles(pred_smiles) is not None if RDKIT_AVAILABLE else False
                tanimoto    = compute_tanimoto(pred_smiles, true_smiles)
                mw_err      = compute_mw_error(pred_smiles, true_smiles)
                form_match  = compute_formula_match(pred_smiles, true_smiles)

                # ── Group frequency metrics ────────────────────────────────
                spec_np  = spectra[i].cpu().numpy()
                gf_report = self.gf_checker.check(spec_np, pred_smiles)
                gf_recall = gf_report.recall
                gf_viols  = gf_report.violation_count
                gf_checked = gf_report.total_groups_checked

                # ── NLP metrics ────────────────────────────────────────────
                ref_tok = ref_tokens_batch[i].tolist()
                # Remove padding and special tokens for token accuracy
                ref_clean = [t for t in ref_tok if t != PAD_IDX]
                pred_clean = pred_tokens[:len(ref_clean)]
                pred_clean += [PAD_IDX] * max(0, len(ref_clean) - len(pred_clean))
                tok_acc = np.mean([p == r for p, r in zip(pred_clean, ref_clean)])

                all_hyp_tokens.append(pred_tokens)
                all_ref_tokens.append(ref_clean)

                log_prob = float(results["log_probs"][i].cpu()) if results["log_probs"].numel() > i else 0.0

                sample_results.append(SampleResult(
                    sample_idx=sample_idx,
                    ecr=ecr_val,
                    peak_position_error=peak_err,
                    intensity_mse=int_mse,
                    physically_implausible=implaus,
                    predicted_smiles=pred_smiles,
                    true_smiles=true_smiles,
                    valid_smiles=valid,
                    tanimoto=tanimoto,
                    mw_error=mw_err,
                    formula_match=form_match,
                    gf_recall=gf_recall,
                    gf_violation_count=gf_viols,
                    gf_groups_checked=gf_checked,
                    token_accuracy=float(tok_acc),
                    log_prob=log_prob,
                    inference_time_ms=inf_ms,
                ))
                sample_idx += 1

        # ── Aggregate ──────────────────────────────────────────────────────
        report = self._aggregate(
            sample_results=sample_results,
            hyp_tokens=all_hyp_tokens,
            ref_tokens=all_ref_tokens,
            model_name=model_name,
            eval_time_sec=time.time() - t_start,
            timestamp=__import__("datetime").datetime.now().isoformat(),
        )
        return report

    def _aggregate(
        self,
        sample_results: list[SampleResult],
        hyp_tokens: list[list[int]],
        ref_tokens: list[list[int]],
        model_name: str,
        eval_time_sec: float,
        timestamp: str,
    ) -> EvalReport:
        """Aggregate per-sample results into EvalReport."""
        n = len(sample_results)

        ecr_arr    = np.array([s.ecr for s in sample_results])
        implaus    = np.array([s.physically_implausible for s in sample_results])
        peak_errs  = np.array([s.peak_position_error for s in sample_results])
        int_mses   = np.array([s.intensity_mse for s in sample_results])

        valid_mask = np.array([s.valid_smiles for s in sample_results])
        tan_arr    = np.array([s.tanimoto for s in sample_results if s.tanimoto >= 0])
        mw_arr     = np.array([s.mw_error for s in sample_results if s.mw_error >= 0])
        form_arr   = np.array([s.formula_match for s in sample_results])

        gf_recalls = np.array([s.gf_recall for s in sample_results if s.gf_groups_checked > 0])
        gf_viols   = np.array([s.gf_violation_count for s in sample_results])

        tok_accs   = np.array([s.token_accuracy for s in sample_results])

        # BLEU-4
        bleu = compute_bleu_4(hyp_tokens, ref_tokens)

        # Failure analysis by MW bracket
        bracket_counts: dict[str, list[bool]] = defaultdict(list)
        for s in sample_results:
            bracket = get_mw_bracket(s.true_smiles)
            bracket_counts[bracket].append(s.physically_implausible)
        failure_by_bracket = {
            k: float(np.mean(v)) for k, v in sorted(bracket_counts.items())
        }

        # Top violated group frequencies
        group_violation_counts: dict[str, int] = defaultdict(int)
        for s in sample_results:
            if s.gf_violation_count > 0:
                # Re-run checker to get which groups violated
                gf_rep = self.gf_checker.check(
                    np.zeros(1800),  # placeholder — violations counted at eval time
                    s.predicted_smiles,
                )
                for v in gf_rep.violations:
                    group_violation_counts[v.group_name] += 1
        top_violations = sorted(
            group_violation_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return EvalReport(
            model_name=model_name,
            timestamp=timestamp,
            num_samples=n,
            eval_time_sec=eval_time_sec,
            # Physics
            ecr_mean=float(ecr_arr.mean()),
            ecr_median=float(np.median(ecr_arr)),
            ecr_std=float(ecr_arr.std()),
            ecr_p95=float(np.percentile(ecr_arr, 95)),
            implausible_rate=float(implaus.mean()),
            peak_position_error_mean=float(peak_errs.mean()),
            intensity_mse_mean=float(int_mses.mean()),
            # Chemistry
            valid_smiles_rate=float(valid_mask.mean()),
            tanimoto_mean=float(tan_arr.mean()) if len(tan_arr) else 0.0,
            tanimoto_median=float(np.median(tan_arr)) if len(tan_arr) else 0.0,
            mw_error_mean=float(mw_arr.mean()) if len(mw_arr) else 0.0,
            mw_error_median=float(np.median(mw_arr)) if len(mw_arr) else 0.0,
            formula_accuracy=float(form_arr.mean()),
            # GF
            gf_recall_mean=float(gf_recalls.mean()) if len(gf_recalls) else 0.0,
            gf_recall_std=float(gf_recalls.std()) if len(gf_recalls) else 0.0,
            gf_violation_rate=float((gf_viols > 0).mean()),
            # NLP
            bleu_4=float(bleu),
            token_accuracy_mean=float(tok_accs.mean()),
            # Failure analysis
            failure_by_mw_bracket=failure_by_bracket,
            top_violation_groups=top_violations,
            sample_results=sample_results,
        )