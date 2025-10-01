"""
spectralm/data/augmentation.py

Spectral data augmentation pipeline.

IR spectra in the wild differ from idealised NIST database entries due to:
  - Baseline drift       (instrument zero-point instability)
  - Gaussian noise       (detector noise)
  - Solvent interference (broad absorptions masking analyte peaks)
  - Concentration shift  (Beer–Lambert amplitude scaling)
  - Water vapour         (sharp spikes at ~1600 and 3700 cm⁻¹)
  - Resolution loss      (Gaussian smoothing of sharp peaks)

Augmentation teaches the model to be robust to these real-world artefacts.
Without augmentation the model learns to rely on absolute peak heights —
which vary between instruments — rather than relative peak patterns.

Key finding from ablation study (Table 1, row: No data augmentation):
    Without augmentation: BLEU drops 0.069 (0.612 → 0.543)
    The model overfits to the clean NIST spectral profiles.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass


# ── Water vapour interference regions (cm⁻¹) ─────────────────────────────────
WATER_VAPOUR_REGIONS = [(1550, 1650), (3550, 3750)]

# ── Common solvent interference regions ──────────────────────────────────────
SOLVENT_REGIONS: dict[str, list[tuple[int, int]]] = {
    "KBr":    [],
    "CCl4":   [(700, 820), (1500, 1600)],
    "CHCl3":  [(660, 780), (1190, 1250)],
    "DMSO":   [(900, 1100), (2900, 3050)],
    "Nujol":  [(720, 740), (1370, 1390), (2850, 2960)],
}

# Standard wavenumber axis (400–4000 cm⁻¹, 1800 points)
WAVENUMBER_AXIS = np.linspace(400.0, 4000.0, 1800)


@dataclass
class AugmentationConfig:
    # Baseline drift
    baseline_drift_prob: float = 0.6
    baseline_drift_max: float = 0.15
    baseline_poly_degree: int = 3

    # Gaussian noise
    noise_prob: float = 0.8
    noise_std_range: tuple[float, float] = (0.005, 0.025)

    # Solvent interference
    solvent_prob: float = 0.35
    solvent_intensity_range: tuple[float, float] = (0.1, 0.6)

    # Concentration scaling (Beer–Lambert amplitude shift)
    concentration_scale_prob: float = 0.5
    concentration_scale_range: tuple[float, float] = (0.6, 1.4)

    # Water vapour contamination
    water_vapour_prob: float = 0.2
    water_vapour_intensity: float = 0.08

    # Spectral resolution degradation (Gaussian smoothing)
    smooth_prob: float = 0.25
    smooth_sigma_range: tuple[float, float] = (0.5, 2.0)


class SpectralAugmentor:
    """
    Applies randomised augmentations to IR spectra during training.

    Augmentations are applied independently per sample in a batch.
    All operations preserve the [0, 1] absorbance range via clipping.

    Usage:
        augmentor = SpectralAugmentor(AugmentationConfig())
        augmented = augmentor(spectrum_tensor)   # (B, W) or (W,)

    Design note:
        Augmentations are applied in a fixed order chosen to simulate
        the physical measurement process:
            1. Concentration scaling  (sample preparation)
            2. Baseline drift         (instrument instability)
            3. Noise                  (detector noise)
            4. Solvent interference   (sample preparation artefact)
            5. Water vapour           (atmospheric contamination)
            6. Smoothing              (resolution degradation)
    """

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations. Input/output shape: (..., W=1800).
        Batched (B, W) or single (W,) both supported.
        """
        was_batched = spectrum.dim() == 2
        if not was_batched:
            spectrum = spectrum.unsqueeze(0)

        B, W = spectrum.shape
        aug = spectrum.clone()

        for i in range(B):
            s = aug[i].numpy().copy()
            s = self._apply_concentration_scale(s)
            s = self._apply_baseline_drift(s)
            s = self._apply_noise(s)
            s = self._apply_solvent(s)
            s = self._apply_water_vapour(s)
            s = self._apply_smoothing(s)
            s = np.clip(s, 0.0, 1.0)
            aug[i] = torch.from_numpy(s.astype(np.float32))

        return aug if was_batched else aug.squeeze(0)

    # ── Individual augmentations ───────────────────────────────────────────

    def _apply_concentration_scale(self, s: np.ndarray) -> np.ndarray:
        """Beer–Lambert amplitude shift: A ∝ concentration."""
        if np.random.rand() < self.config.concentration_scale_prob:
            lo, hi = self.config.concentration_scale_range
            s = s * np.random.uniform(lo, hi)
        return s

    def _apply_baseline_drift(self, s: np.ndarray) -> np.ndarray:
        """Polynomial baseline drift from instrument instability."""
        if np.random.rand() < self.config.baseline_drift_prob:
            x = np.linspace(0, 1, len(s))
            degree = self.config.baseline_poly_degree
            coeffs = np.random.randn(degree + 1) * self.config.baseline_drift_max
            drift = np.polyval(coeffs, x)
            drift -= drift.mean()   # zero-mean so signal is preserved
            s = s + drift
        return s

    def _apply_noise(self, s: np.ndarray) -> np.ndarray:
        """Gaussian detector noise."""
        if np.random.rand() < self.config.noise_prob:
            lo, hi = self.config.noise_std_range
            std = np.random.uniform(lo, hi)
            s = s + np.random.normal(0, std, size=s.shape)
        return s

    def _apply_solvent(self, s: np.ndarray) -> np.ndarray:
        """
        Solvent/mulling agent interference.
        Adds broad absorptions in solvent-characteristic regions.
        Most common: Nujol (mineral oil) in 2850–2960 cm⁻¹.
        """
        if np.random.rand() < self.config.solvent_prob:
            solvent = np.random.choice(list(SOLVENT_REGIONS.keys()))
            regions = SOLVENT_REGIONS[solvent]
            lo_int, hi_int = self.config.solvent_intensity_range
            for wn_lo, wn_hi in regions:
                mask = (WAVENUMBER_AXIS >= wn_lo) & (WAVENUMBER_AXIS <= wn_hi)
                if mask.sum() == 0:
                    continue
                intensity = np.random.uniform(lo_int, hi_int)
                s[mask] = np.maximum(
                    s[mask], intensity * np.random.rand(mask.sum())
                )
        return s

    def _apply_water_vapour(self, s: np.ndarray) -> np.ndarray:
        """
        Atmospheric water vapour contamination.
        Produces sharp spikes in the 1550–1650 and 3550–3750 cm⁻¹ regions.
        """
        if np.random.rand() < self.config.water_vapour_prob:
            for wn_lo, wn_hi in WATER_VAPOUR_REGIONS:
                mask = (WAVENUMBER_AXIS >= wn_lo) & (WAVENUMBER_AXIS <= wn_hi)
                idxs = np.where(mask)[0]
                if len(idxs) < 5:
                    continue
                spike_pos = np.random.choice(idxs, size=5, replace=False)
                s[spike_pos] = np.minimum(
                    1.0,
                    s[spike_pos] + self.config.water_vapour_intensity
                )
        return s

    def _apply_smoothing(self, s: np.ndarray) -> np.ndarray:
        """
        Gaussian smoothing to simulate lower instrument resolution.
        Broadens sharp peaks — common in older ATR-FTIR instruments.
        """
        if np.random.rand() < self.config.smooth_prob:
            from scipy.ndimage import gaussian_filter1d
            lo, hi = self.config.smooth_sigma_range
            sigma = np.random.uniform(lo, hi)
            s = gaussian_filter1d(s, sigma=sigma)
        return s