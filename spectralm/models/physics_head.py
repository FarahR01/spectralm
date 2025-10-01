"""
spectralm/models/physics_head.py

Physics head: reconstructs a predicted IR spectrum from the model's
predicted group activations using Lorentzian peak shapes.

This is the "forward physics model" — given what the Transformer thinks
the molecule is (via group logits), reconstruct what the IR spectrum
SHOULD look like according to Beer–Lambert law.

The BeerLambertConstraint then compares this reconstruction to the actual
input spectrum, closing the physics feedback loop.

Physical motivation:
    IR absorption bands are Lorentzian-shaped due to homogeneous
    broadening from molecular collisions (lifetime broadening):

        A(ν) = I_max / (1 + ((ν - ν_0) / (Δν/2))²)

    where:
        ν_0   = peak centre (cm⁻¹)   — constrained to group's expected range
        Δν    = FWHM                  — learned per group
        I_max = peak intensity        — from group_logits via sigmoid

    Additivity of absorbances (Beer–Lambert):
        A_total(ν) = Σ_g A_g(ν)

    This allows the reconstruction to be differentiable end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from spectralm.physics.group_frequencies import GROUP_FREQUENCY_TABLE

# Standard wavenumber axis (400–4000 cm⁻¹, 1800 points)
WAVENUMBER_AXIS_NP = np.linspace(400.0, 4000.0, 1800)


class LorentzianPeakModel(nn.Module):
    """
    Reconstructs an IR spectrum from predicted group intensities
    using learnable Lorentzian peak shapes.

    Learnable parameters per functional group:
        peak_centre  : constrained to group's physical range via sigmoid
        log_half_width: log(FWHM/2), constrained to [5, 300] cm⁻¹

    Fixed inputs per group:
        group_intensities : (B, G) from sigmoid(group_logits)

    Output:
        reconstructed_spectrum : (B, W) absorbance values in [0, 1]
    """

    def __init__(self, num_groups: int, wavenumber_steps: int = 1800):
        super().__init__()
        self.num_groups       = num_groups
        self.wavenumber_steps = wavenumber_steps

        wn = torch.from_numpy(WAVENUMBER_AXIS_NP).float()
        self.register_buffer("wavenumbers", wn)   # (W,)

        # Initialise peak centres to group midpoints
        centres = torch.tensor([
            (gf.low + gf.high) / 2.0 for gf in GROUP_FREQUENCY_TABLE
        ])  # (G,)
        self.peak_centres = nn.Parameter(centres)

        # Initialise half-widths to quarter of group range
        half_widths = torch.tensor([
            max((gf.high - gf.low) / 4.0, 5.0)
            for gf in GROUP_FREQUENCY_TABLE
        ])
        self.log_half_widths = nn.Parameter(torch.log(half_widths))

        # Physical range constraints for peak centres
        low_bounds  = torch.tensor([gf.low  for gf in GROUP_FREQUENCY_TABLE])
        high_bounds = torch.tensor([gf.high for gf in GROUP_FREQUENCY_TABLE])
        self.register_buffer("low_bounds",  low_bounds)
        self.register_buffer("high_bounds", high_bounds)

    def forward(self, group_intensities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            group_intensities : (B, G) predicted intensity per functional group,
                                       from sigmoid(group_logits)
        Returns:
            reconstructed_spectrum : (B, W) absorbance values in [0, 1]
        """
        # Constrain centres to physical ranges
        centres = self.low_bounds + torch.sigmoid(self.peak_centres) * (
            self.high_bounds - self.low_bounds
        )   # (G,)

        half_widths = torch.exp(self.log_half_widths).clamp(5.0, 300.0)  # (G,)

        # Lorentzian shape: (G, W)
        wn  = self.wavenumbers.unsqueeze(0)   # (1, W)
        c   = centres.unsqueeze(1)             # (G, 1)
        hw  = half_widths.unsqueeze(1)         # (G, 1)

        lorentzians = 1.0 / (1.0 + ((wn - c) / hw) ** 2)   # (G, W)

        # Scale by predicted intensities and sum (Beer–Lambert additivity)
        # group_intensities: (B, G) → (B, G, 1)
        # lorentzians:       (G, W) → (1, G, W)
        scaled  = group_intensities.unsqueeze(2) * lorentzians.unsqueeze(0)
        summed  = scaled.sum(dim=1).clamp(0.0, 1.0)   # (B, W)

        return summed


class PhysicsHead(nn.Module):
    """
    Combines group logits → group probabilities → Lorentzian spectrum reconstruction.

    Sits between the encoder (which produces group_logits) and the
    BeerLambertConstraint (which compares reconstructed to observed).

    This is the module that makes SpectraLM's physics constraint differentiable:
        encoder.group_head → PhysicsHead → BeerLambertConstraint → loss

    Returns:
        group_probs         : (B, G) sigmoid group probabilities
        reconstructed_spec  : (B, W) Lorentzian sum spectrum
    """

    def __init__(self, num_groups: int = 22):
        super().__init__()
        self.peak_model = LorentzianPeakModel(num_groups)

    def forward(
        self,
        group_logits: torch.Tensor,    # (B, G)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_probs       = torch.sigmoid(group_logits)
        reconstructed_spec = self.peak_model(group_probs)
        return group_probs, reconstructed_spec