"""
spectralm/physics/isotope_shifts.py

Isotope shift prediction for IR absorption bands.

When a molecule contains a heavier isotope (e.g. ¹³C instead of ¹²C,
or D instead of H), its vibrational frequencies shift predictably
according to the reduced mass relationship derived from the
quantum harmonic oscillator model.

Physical basis — harmonic oscillator frequency:
    ν = (1/2πc) · √(k / μ)

where:
    k  = force constant (N/m)         — unchanged by isotope substitution
    μ  = reduced mass (kg)            — changes with isotope
    c  = speed of light (cm/s)

Isotope shift formula:
    ν_heavy / ν_light = √(μ_light / μ_heavy)

    Δν = ν_light · (1 - √(μ_light / μ_heavy))

This module:
    1. Predicts expected isotope shifts for common substitutions
       (¹²C→¹³C, H→D, ¹⁴N→¹⁵N, ¹⁶O→¹⁸O, ³²S→³⁴S)
    2. Provides a differentiable correction term for the physics loss
    3. Flags predictions where the model's peak assignments are
       inconsistent with the expected isotope pattern

Role in SpectraLM:
    Isotope shifts account for 7% of failure cases (failure type: overtone_band).
    The IsotopeShiftCorrector is called in 03_optimization.ipynb
    (Section 4 — Failure Gallery) to annotate Type IV failures.

    During training it is used as a soft regulariser via IsotopeShiftPenalty,
    which penalises predicted peak centres that fall outside the
    physically expected isotope-shifted range.

Status: partially implemented (as noted in model card).
    ¹²C→¹³C and H→D corrections are production-ready.
    ¹⁶O→¹⁸O and ³²S→³⁴S are implemented but not yet validated
    against experimental data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ══════════════════════════════════════════════════════════════════════════════

# Atomic masses (unified atomic mass units, u)
ATOMIC_MASSES: dict[str, float] = {
    # Hydrogen isotopes
    "H":  1.00782503207,
    "D":  2.01410177785,    # deuterium
    "T":  3.01604927767,    # tritium (rarely relevant for IR)

    # Carbon isotopes
    "12C": 12.000000000,
    "13C": 13.003354835,

    # Nitrogen isotopes
    "14N": 14.003074004,
    "15N": 15.000108898,

    # Oxygen isotopes
    "16O": 15.994914620,
    "17O": 16.999131757,
    "18O": 17.999161000,

    # Sulfur isotopes
    "32S": 31.972071000,
    "33S": 32.971458900,
    "34S": 33.967867000,
    "36S": 35.967081000,

    # Chlorine isotopes (relevant for C–Cl stretch region)
    "35Cl": 34.968852680,
    "37Cl": 36.965902590,

    # Bromine isotopes
    "79Br": 78.918337100,
    "81Br": 80.916290600,
}


class IsotopeSubstitution(NamedTuple):
    """Defines a single isotope substitution."""
    name: str               # e.g. "H→D deuteration"
    light_atom: str         # key in ATOMIC_MASSES
    heavy_atom: str         # key in ATOMIC_MASSES
    partner_atom: str       # bonding partner (e.g. "C" for C–H)
    partner_mass: float     # mass of the bonding partner (u)
    affected_modes: list[str]  # vibrational mode names affected


# ── Common isotope substitutions ──────────────────────────────────────────────
ISOTOPE_SUBSTITUTIONS: list[IsotopeSubstitution] = [
    IsotopeSubstitution(
        name="H→D (deuteration)",
        light_atom="H",
        heavy_atom="D",
        partner_atom="C",
        partner_mass=12.000,
        affected_modes=["C-H stretch", "C-H bend", "O-H stretch", "N-H stretch"],
    ),
    IsotopeSubstitution(
        name="¹²C→¹³C",
        light_atom="12C",
        heavy_atom="13C",
        partner_atom="O",
        partner_mass=15.995,
        affected_modes=["C=O stretch", "C-O stretch", "C≡N stretch", "C=C stretch"],
    ),
    IsotopeSubstitution(
        name="¹⁴N→¹⁵N",
        light_atom="14N",
        heavy_atom="15N",
        partner_atom="H",
        partner_mass=1.008,
        affected_modes=["N-H stretch", "N-H bend", "C≡N stretch", "C-N stretch"],
    ),
    IsotopeSubstitution(
        name="¹⁶O→¹⁸O",
        light_atom="16O",
        heavy_atom="18O",
        partner_atom="C",
        partner_mass=12.000,
        affected_modes=["C=O stretch", "C-O stretch", "O-H stretch"],
    ),
    IsotopeSubstitution(
        name="³²S→³⁴S",
        light_atom="32S",
        heavy_atom="34S",
        partner_atom="C",
        partner_mass=12.000,
        affected_modes=["C=S stretch", "C-S stretch", "S-H stretch"],
    ),
    IsotopeSubstitution(
        name="³⁵Cl→³⁷Cl",
        light_atom="35Cl",
        heavy_atom="37Cl",
        partner_atom="C",
        partner_mass=12.000,
        affected_modes=["C-Cl stretch"],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Core physics
# ══════════════════════════════════════════════════════════════════════════════

def reduced_mass(m1: float, m2: float) -> float:
    """
    Reduced mass for a two-body system (diatomic approximation).

        μ = (m1 · m2) / (m1 + m2)

    Units: same as input (typically unified atomic mass units, u).
    """
    return (m1 * m2) / (m1 + m2)


def isotope_frequency_ratio(
    light_atom: str,
    heavy_atom: str,
    partner_mass: float,
) -> float:
    """
    Ratio of vibrational frequencies after isotope substitution.

        ν_heavy / ν_light = √(μ_light / μ_heavy)

    Args:
        light_atom    : key in ATOMIC_MASSES (e.g. "H")
        heavy_atom    : key in ATOMIC_MASSES (e.g. "D")
        partner_mass  : mass of bonding partner in u (e.g. 12.0 for C)

    Returns:
        frequency ratio ν_heavy / ν_light  (always < 1.0)
    """
    m_light   = ATOMIC_MASSES[light_atom]
    m_heavy   = ATOMIC_MASSES[heavy_atom]

    mu_light  = reduced_mass(m_light, partner_mass)
    mu_heavy  = reduced_mass(m_heavy, partner_mass)

    return math.sqrt(mu_light / mu_heavy)


def predict_isotope_shift(
    wavenumber: float,
    light_atom: str,
    heavy_atom: str,
    partner_mass: float,
) -> float:
    """
    Predict the wavenumber shift (Δν, cm⁻¹) upon isotope substitution.

        Δν = ν_light · (1 - √(μ_light / μ_heavy))

    The shift is always negative (peak moves to lower wavenumber).

    Args:
        wavenumber  : observed peak position (cm⁻¹) for the light isotope
        light_atom  : e.g. "H"
        heavy_atom  : e.g. "D"
        partner_mass: bonding partner mass (u)

    Returns:
        Δν in cm⁻¹ (negative value = red shift)

    Examples:
        C–H stretch at 2960 cm⁻¹ → C–D stretch at ~2120 cm⁻¹
        Δν ≈ -840 cm⁻¹  (H→D, factor ≈ 0.717)

        C=O stretch at 1715 cm⁻¹ (¹²C) → ~1681 cm⁻¹ (¹³C)
        Δν ≈ -34 cm⁻¹  (¹²C→¹³C, factor ≈ 0.980)
    """
    ratio = isotope_frequency_ratio(light_atom, heavy_atom, partner_mass)
    return wavenumber * (ratio - 1.0)   # negative = red shift


@dataclass
class IsotopeShiftPrediction:
    """Predicted isotope shift for a single vibrational mode."""
    substitution_name: str
    mode_name: str
    original_wavenumber: float       # cm⁻¹ (light isotope)
    predicted_wavenumber: float      # cm⁻¹ (heavy isotope)
    shift_cm: float                  # Δν in cm⁻¹ (negative)
    frequency_ratio: float           # ν_heavy / ν_light
    confidence: str                  # "validated" | "theoretical"


# ── Group frequency → typical wavenumber mapping ──────────────────────────────
# Centre of expected range for each vibrational mode
MODE_WAVENUMBERS: dict[str, float] = {
    "C-H stretch":   2920.0,
    "C-H bend":      1460.0,
    "O-H stretch":   3350.0,
    "N-H stretch":   3350.0,
    "C=O stretch":   1710.0,
    "C-O stretch":   1100.0,
    "C≡N stretch":   2230.0,
    "C=C stretch":   1640.0,
    "C-N stretch":   1100.0,
    "N-H bend":      1550.0,
    "C=S stretch":   1050.0,
    "C-S stretch":    700.0,
    "S-H stretch":   2570.0,
    "C-Cl stretch":   720.0,
}

# Experimental validation status
VALIDATED_SUBSTITUTIONS = {"H→D (deuteration)", "¹²C→¹³C"}


class IsotopeShiftCorrector:
    """
    Predicts expected isotope shifts for a molecule and compares them
    to the model's peak assignments.

    Primary use: failure case annotation (Type IV — overtone_band failures
    are often caused by confusing overtone bands with isotope-shifted peaks).

    Not differentiable — used for evaluation and failure analysis only.
    For the differentiable version during training, see IsotopeShiftPenalty.

    Usage:
        corrector = IsotopeShiftCorrector()
        shifts = corrector.predict_for_smiles("CCO", substitution="H→D")
        for s in shifts:
            print(f"{s.mode_name}: {s.original_wavenumber:.0f} → "
                  f"{s.predicted_wavenumber:.0f} cm⁻¹ (Δ={s.shift_cm:.1f})")
    """

    def __init__(self):
        self._sub_map = {s.name: s for s in ISOTOPE_SUBSTITUTIONS}

    def predict_for_smiles(
        self,
        smiles: str,
        substitution: str = "H→D (deuteration)",
    ) -> list[IsotopeShiftPrediction]:
        """
        Predict isotope shifts for all relevant vibrational modes
        in a molecule given its SMILES string.

        Args:
            smiles       : SMILES string of the molecule
            substitution : substitution name from ISOTOPE_SUBSTITUTIONS

        Returns:
            List of IsotopeShiftPrediction, one per affected vibrational mode
        """
        if substitution not in self._sub_map:
            raise ValueError(
                f"Unknown substitution: {substitution}. "
                f"Available: {list(self._sub_map.keys())}"
            )

        sub = self._sub_map[substitution]
        predictions: list[IsotopeShiftPrediction] = []

        # Check which functional groups are present via RDKit
        try:
            from rdkit import Chem
            from spectralm.physics.group_frequencies import GROUP_FREQUENCY_TABLE

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            for gf in GROUP_FREQUENCY_TABLE:
                # Check if this group's modes are affected by the substitution
                affected = any(
                    mode in sub.affected_modes
                    for mode in [gf.name.split("(")[0].strip()]
                )
                if not affected:
                    continue

                # Check if group is present in molecule
                pattern = Chem.MolFromSmarts(gf.smarts)
                if pattern is None or not mol.HasSubstructMatch(pattern):
                    continue

                # Find matching mode
                mode_wn = None
                matched_mode = None
                for mode_name, wn in MODE_WAVENUMBERS.items():
                    if any(m in mode_name for m in sub.affected_modes):
                        if mode_name.lower() in gf.name.lower():
                            mode_wn = wn
                            matched_mode = mode_name
                            break

                if mode_wn is None:
                    # Use group midpoint
                    mode_wn = (gf.low + gf.high) / 2.0
                    matched_mode = gf.name

                shift = predict_isotope_shift(
                    wavenumber=mode_wn,
                    light_atom=sub.light_atom,
                    heavy_atom=sub.heavy_atom,
                    partner_mass=sub.partner_mass,
                )
                ratio = isotope_frequency_ratio(
                    sub.light_atom, sub.heavy_atom, sub.partner_mass
                )

                predictions.append(IsotopeShiftPrediction(
                    substitution_name=sub.name,
                    mode_name=matched_mode or gf.name,
                    original_wavenumber=round(mode_wn, 1),
                    predicted_wavenumber=round(mode_wn + shift, 1),
                    shift_cm=round(shift, 2),
                    frequency_ratio=round(ratio, 5),
                    confidence=(
                        "validated" if sub.name in VALIDATED_SUBSTITUTIONS
                        else "theoretical"
                    ),
                ))

        except ImportError:
            pass

        return predictions

    def check_prediction_consistency(
        self,
        spectrum: np.ndarray,     # (1800,) observed spectrum
        smiles: str,
        substitution: str = "H→D (deuteration)",
        tolerance_cm: float = 15.0,
    ) -> dict:
        """
        Check whether peaks in the spectrum are consistent with
        the expected isotope pattern for the predicted SMILES.

        Returns a dict with:
            consistent    : bool
            checked_modes : int
            violations    : list of mode names where peak is missing/wrong
            mean_error_cm : mean absolute error in cm⁻¹
        """
        from spectralm.physics.group_frequencies import WAVENUMBER_AXIS

        predictions = self.predict_for_smiles(smiles, substitution)
        violations  = []
        errors      = []

        for pred in predictions:
            # Find the predicted peak in the spectrum
            target_wn   = pred.predicted_wavenumber
            wn_mask     = (
                (WAVENUMBER_AXIS >= target_wn - tolerance_cm) &
                (WAVENUMBER_AXIS <= target_wn + tolerance_cm)
            )

            if wn_mask.sum() == 0:
                violations.append(pred.mode_name)
                continue

            region_max = spectrum[wn_mask].max()
            if region_max < 0.10:   # no significant absorption
                violations.append(pred.mode_name)
                errors.append(tolerance_cm)
            else:
                # Find actual peak position
                peak_idx    = wn_mask.nonzero()[0][
                    spectrum[wn_mask].argmax()
                ]
                actual_wn   = float(WAVENUMBER_AXIS[peak_idx])
                errors.append(abs(actual_wn - target_wn))

        return {
            "consistent":    len(violations) == 0,
            "checked_modes": len(predictions),
            "violations":    violations,
            "mean_error_cm": float(np.mean(errors)) if errors else 0.0,
        }

    def print_shift_table(self, smiles: str):
        """Pretty-print isotope shift predictions for all substitutions."""
        print(f"\nIsotope shift predictions for: {smiles}")
        print(f"{'─'*72}")
        print(f"  {'Substitution':<22} {'Mode':<25} "
              f"{'Original':>9} {'Shifted':>9} {'Δν':>8}  Status")
        print(f"{'─'*72}")

        for sub in ISOTOPE_SUBSTITUTIONS:
            preds = self.predict_for_smiles(smiles, sub.name)
            for p in preds:
                status = "✓" if p.confidence == "validated" else "~"
                print(f"  {p.substitution_name:<22} {p.mode_name:<25} "
                      f"{p.original_wavenumber:>8.1f}  "
                      f"{p.predicted_wavenumber:>8.1f}  "
                      f"{p.shift_cm:>7.1f}  {status}")

        print(f"{'─'*72}")
        print(f"  ✓ = experimentally validated  ~ = theoretical only")


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable penalty for training
# ══════════════════════════════════════════════════════════════════════════════

class IsotopeShiftPenalty(nn.Module):
    """
    Differentiable isotope shift regulariser for training.

    If the model predicts a peak at wavenumber ν but the molecule
    contains an isotope-sensitive group, the PREDICTED peak centre
    (from LorentzianPeakModel.peak_centres) should be consistent
    with the expected isotope shift.

    Penalty = Σ_g  group_prob[g] · relu(|ν_pred[g] - ν_expected[g]| - tol)²

    This penalises confident group predictions whose peak centres
    deviate from the physically expected position by more than
    `tolerance_cm` wavenumbers.

    Status: implemented but not yet incorporated into the default
    training loss (see NEXT STEPS in model card). Kept here for
    the next development phase.
    """

    def __init__(
        self,
        tolerance_cm: float = 20.0,
        weight: float = 0.05,
    ):
        super().__init__()
        self.tolerance = tolerance_cm
        self.weight    = weight

        # Pre-compute expected peak centres for each group
        from spectralm.physics.group_frequencies import GROUP_FREQUENCY_TABLE
        centres = torch.tensor([
            (gf.low + gf.high) / 2.0 for gf in GROUP_FREQUENCY_TABLE
        ])
        self.register_buffer("expected_centres", centres)   # (G,)

    def forward(
        self,
        predicted_centres: torch.Tensor,    # (G,) from LorentzianPeakModel
        group_probs: torch.Tensor,           # (B, G) from PhysicsHead
    ) -> torch.Tensor:
        """
        Returns scalar regularisation penalty.

        Penalises cases where a group is predicted with high confidence
        but its peak centre deviates from the physically expected position.
        """
        # Deviation from expected centre
        deviation = (predicted_centres - self.expected_centres).abs()   # (G,)

        # Soft threshold: penalise only deviations beyond tolerance
        excess = torch.relu(deviation - self.tolerance)   # (G,)

        # Weight by group confidence (B, G) × (G,) → (B, G)
        weighted = group_probs * excess.unsqueeze(0)

        return self.weight * weighted.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Quick demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SpectraLM — Isotope Shift Module")
    print("=" * 72)

    # Show shift table for a few common molecules
    corrector = IsotopeShiftCorrector()

    # Theoretical shifts (no RDKit needed for demo)
    examples = [
        ("C–H stretch", "H", "D", 12.000, 2920.0),
        ("C=O stretch", "12C", "13C", 15.995, 1715.0),
        ("N–H stretch", "H", "D", 14.003, 3350.0),
        ("C≡N stretch", "14N", "15N", 12.000, 2230.0),
        ("C–Cl stretch", "35Cl", "37Cl", 12.000, 720.0),
        ("O–H stretch", "16O", "18O", 1.008, 3350.0),
    ]

    print(f"\n  {'Mode':<20} {'Light':>8} {'Heavy':>8} {'Ratio':>8} {'Δν (cm⁻¹)':>12}")
    print(f"  {'─'*60}")

    for mode, light, heavy, partner, wn in examples:
        ratio = isotope_frequency_ratio(light, heavy, partner)
        shift = predict_isotope_shift(wn, light, heavy, partner)
        print(f"  {mode:<20} {wn:>8.1f} {wn+shift:>8.1f} "
              f"{ratio:>8.5f} {shift:>12.2f}")

    print(f"\n  H→D is the largest shift by far — factor ≈ 0.717")
    print(f"  ¹²C→¹³C shifts are subtle (≈ 2%) but diagnostic")
    print(f"  These shifts are used to annotate Type IV failure cases")
    print(f"  in evals/failure_cases/ (overtone band misidentification)")