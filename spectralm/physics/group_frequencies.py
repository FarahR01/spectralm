"""
Group frequency table and violation detector.

Each functional group has a known characteristic absorption range (cm⁻¹).
If the model predicts a molecule containing a functional group but the
corresponding spectral region shows no absorption, this is a
group frequency violation — a direct physics error.

Reference: Silverstein, Webster & Kiemle, "Spectrometric Identification
of Organic Compounds", 8th ed., Table 3-1.
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple


class GroupFrequencyRange(NamedTuple):
    name: str
    low: float          # cm⁻¹ lower bound
    high: float         # cm⁻¹ upper bound
    min_intensity: float  # minimum expected absorbance (0–1 scale)
    smarts: str         # SMARTS pattern for group detection


# ── Canonical group frequency table ──────────────────────────────────────────
# Format: (name, low_cm⁻¹, high_cm⁻¹, min_absorbance, SMARTS)
GROUP_FREQUENCY_TABLE: list[GroupFrequencyRange] = [
    GroupFrequencyRange("O-H stretch (alcohol)",    3200, 3550, 0.40, "[OX2H]"),
    GroupFrequencyRange("O-H stretch (carboxylic)", 2500, 3300, 0.30, "[CX3](=O)[OX2H1]"),
    GroupFrequencyRange("N-H stretch (primary)",    3300, 3500, 0.25, "[NX3;H2]"),
    GroupFrequencyRange("N-H stretch (secondary)",  3100, 3350, 0.20, "[NX3;H1]"),
    GroupFrequencyRange("C-H stretch (alkane)",     2850, 2960, 0.35, "[CX4;H]"),
    GroupFrequencyRange("C-H stretch (alkene)",     3020, 3100, 0.15, "[CX3;H]=[CX3]"),
    GroupFrequencyRange("C-H stretch (aromatic)",   3000, 3100, 0.15, "c[H]"),
    GroupFrequencyRange("C-H stretch (aldehyde)",   2700, 2830, 0.10, "[CX3H1](=O)"),
    GroupFrequencyRange("C≡N stretch (nitrile)",    2200, 2260, 0.30, "[CX2]#[NX1]"),
    GroupFrequencyRange("C≡C stretch (alkyne)",     2100, 2260, 0.10, "[CX2]#[CX2]"),
    GroupFrequencyRange("C=O stretch (ketone)",     1680, 1750, 0.60, "[CX3](=O)[#6]"),
    GroupFrequencyRange("C=O stretch (aldehyde)",   1720, 1740, 0.60, "[CX3H1](=O)"),
    GroupFrequencyRange("C=O stretch (ester)",      1735, 1750, 0.65, "[CX3](=O)[OX2][#6]"),
    GroupFrequencyRange("C=O stretch (amide)",      1630, 1690, 0.55, "[CX3](=O)[NX3]"),
    GroupFrequencyRange("C=O stretch (carboxylic)", 1700, 1725, 0.65, "[CX3](=O)[OX2H1]"),
    GroupFrequencyRange("C=C stretch (alkene)",     1620, 1680, 0.20, "[CX3]=[CX3]"),
    GroupFrequencyRange("C=C stretch (aromatic)",   1450, 1600, 0.30, "c1ccccc1"),
    GroupFrequencyRange("C-O stretch (ether)",      1000, 1150, 0.40, "[CX4][OX2][CX4]"),
    GroupFrequencyRange("C-O stretch (alcohol)",     950, 1150, 0.35, "[CX4][OX2H]"),
    GroupFrequencyRange("N-O stretch (nitro)",      1500, 1570, 0.55, "[NX3](=O)=O"),
    GroupFrequencyRange("C-F stretch",               1000, 1400, 0.50, "[#6][F]"),
    GroupFrequencyRange("C-Cl stretch",               600,  800, 0.45, "[#6][Cl]"),
]

# Build wavenumber index for fast lookup
WAVENUMBER_AXIS = np.linspace(400.0, 4000.0, 1800)


@dataclass
class GroupFrequencyViolation:
    """A single detected group frequency violation."""
    group_name: str
    expected_range: tuple[float, float]
    observed_max_intensity: float
    required_min_intensity: float
    sample_idx: int

    @property
    def severity(self) -> float:
        """How far below threshold the observed intensity is."""
        return max(0.0, self.required_min_intensity - self.observed_max_intensity)


@dataclass
class GroupFrequencyReport:
    violations: list[GroupFrequencyViolation] = field(default_factory=list)
    total_groups_checked: int = 0
    recall: float = 0.0  # fraction of expected groups correctly showing absorbance

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class GroupFrequencyChecker:
    """
    Checks whether a spectrum contains the expected absorptions
    for the functional groups present in a predicted SMILES string.

    Not differentiable — used for evaluation and failure analysis only.
    For the differentiable version, see GroupFrequencyPenalty below.
    """

    def __init__(self, min_intensity_scale: float = 1.0):
        """
        min_intensity_scale: multiplier on required min intensities.
        Set < 1.0 to be more lenient (e.g., dilute samples).
        """
        self.scale = min_intensity_scale

    def check(
        self,
        spectrum: np.ndarray,  # (1800,) absorbance values 0–1
        smiles: str,
    ) -> GroupFrequencyReport:
        """
        For each functional group detected in `smiles`,
        verify the spectrum contains the expected absorption band.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            raise ImportError("RDKit required for group frequency checking.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return GroupFrequencyReport()

        report = GroupFrequencyReport()
        for gf in GROUP_FREQUENCY_TABLE:
            # Check if this group is present in the molecule
            pattern = Chem.MolFromSmarts(gf.smarts)
            if pattern is None:
                continue
            if not mol.HasSubstructMatch(pattern):
                continue

            # Group is present — check spectrum for expected absorption
            report.total_groups_checked += 1
            region_mask = (WAVENUMBER_AXIS >= gf.low) & (WAVENUMBER_AXIS <= gf.high)
            region_spectrum = spectrum[region_mask]

            if len(region_spectrum) == 0:
                continue

            max_intensity = float(region_spectrum.max())
            required = gf.min_intensity * self.scale

            if max_intensity < required:
                # Violation: group present in molecule, missing from spectrum
                # (or vice versa — spectrum shows peak, group not in structure)
                report.violations.append(
                    GroupFrequencyViolation(
                        group_name=gf.name,
                        expected_range=(gf.low, gf.high),
                        observed_max_intensity=max_intensity,
                        required_min_intensity=required,
                        sample_idx=-1,  # set by caller
                    )
                )

        if report.total_groups_checked > 0:
            report.recall = 1.0 - (report.violation_count / report.total_groups_checked)

        return report


class GroupFrequencyPenalty(nn.Module if True else object):
    """
    Differentiable group frequency penalty for training.

    Given a soft group-presence matrix G (B, num_groups) predicted by the model
    and the input spectrum S (B, W), penalises predictions where a group is
    confidently predicted but its expected spectral region has low absorbance.

    penalty = Σ_g  G[:,g] * relu(min_intensity[g] - max(S[:, region_g]))
    """

    def __init__(self, device: str = "cpu"):
        import torch.nn as nn
        super().__init__()

        # Pre-compute region masks as buffers
        wn = torch.from_numpy(WAVENUMBER_AXIS).float()
        masks = []
        min_intensities = []
        for gf in GROUP_FREQUENCY_TABLE:
            mask = ((wn >= gf.low) & (wn <= gf.high)).float()
            masks.append(mask)
            min_intensities.append(gf.min_intensity)

        masks_tensor = torch.stack(masks, dim=0)           # (G, W)
        min_int_tensor = torch.tensor(min_intensities)     # (G,)

        self.register_buffer("region_masks", masks_tensor)
        self.register_buffer("min_intensities", min_int_tensor)
        self.num_groups = len(GROUP_FREQUENCY_TABLE)

    def forward(
        self,
        spectrum: torch.Tensor,        # (B, W)
        group_logits: torch.Tensor,    # (B, G) — model's group presence logits
    ) -> torch.Tensor:
        """Returns scalar penalty loss."""
        import torch.nn as nn
        import torch.nn.functional as F

        group_probs = torch.sigmoid(group_logits)  # (B, G)

        # For each group, find the max absorbance in its expected region
        # spectrum: (B, W), region_masks: (G, W)
        # -> region_spectra: (B, G) — max absorbance per group region per sample
        masked = spectrum.unsqueeze(1) * self.region_masks.unsqueeze(0)  # (B, G, W)
        region_max = masked.max(dim=-1).values  # (B, G)

        # Violation: group predicted with confidence but region is weak
        shortfall = F.relu(self.min_intensities.unsqueeze(0) - region_max)  # (B, G)
        penalty = (group_probs * shortfall).sum(dim=-1).mean()  # scalar

        return penalty


# Make GroupFrequencyPenalty a proper nn.Module
import torch.nn as nn
GroupFrequencyPenalty.__bases__ = (nn.Module,)