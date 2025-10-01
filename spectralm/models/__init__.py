"""
spectralm/models/__init__.py

SpectraLM: full model assembling encoder, transformer, and physics head.

Architecture summary:
    Input spectrum (B, 1800)
        ↓  SpectralEncoder
           1D-CNN (4 layers, stride=2) → (B, 112, 256)
           + WavenumberPositionalEncoding  ← KEY: physical, not token-index
           → spectral tokens (B, 112, 256)
           → group_logits   (B, 22)
        ↓  PhysicsHead
           LorentzianPeakModel → reconstructed_spectrum (B, 1800)
        ↓  BeerLambertConstraint
           ECR loss (physics constraint)
        ↓  GroupFrequencyPenalty
           GF penalty (physics constraint)
        ↓  SpectralTransformer
           Cross-attention encoder-decoder
           → SMILES logits (B, T, vocab_size)

Total loss = CE_loss
           + λ_bl=0.30 · BeerLambert_ECR
           + λ_gf=0.15 · GroupFrequency_penalty

Physics loss annealed from 0 → full value over first 10 epochs.
(See scripts/train.py: physics_lambda_schedule)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from spectralm.models.encoder import SpectralEncoder, EncoderConfig
from spectralm.models.transformer import SpectralTransformer, TransformerConfig
from spectralm.models.physics_head import PhysicsHead
from spectralm.physics.beer_lambert import BeerLambertConstraint, BeerLambertResidual
from spectralm.physics.group_frequencies import GroupFrequencyPenalty


@dataclass
class SpectraLMConfig:
    """
    Full model configuration.

    Key hyperparameters from ablation study (Table 1):
        lambda_beer_lambert = 0.30  — best BLEU AND best ECR simultaneously
        lambda_group_freq   = 0.15  — pure physics gain, near-zero BLEU cost
        use_continuous_pe   = True  — most impactful single choice (+0.300 BLEU)
    """
    encoder:     EncoderConfig     = field(default_factory=EncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # Physics loss weights — confirmed optimal via λ sweep (02_experiments.ipynb)
    lambda_beer_lambert:      float = 0.30
    lambda_group_freq:        float = 0.15
    implausibility_threshold: float = 0.25


class SpectraLM(nn.Module):
    """
    Physics-informed IR spectrum → SMILES/IUPAC name translation model.

    The model cannot learn Beer–Lambert law from data alone — it is a
    physical law. Pretending otherwise produces a model that generates
    chemically fluent but physically nonsensical outputs.
    (See 02_experiments.ipynb, H2 and H3 post-mortems.)

    Usage:
        model  = SpectraLM()
        output = model(spectrum, tgt_tokens)   # training
        pred   = model.predict(spectrum)       # inference
    """

    def __init__(self, config: SpectraLMConfig | None = None):
        super().__init__()
        self.config = config or SpectraLMConfig()

        self.encoder        = SpectralEncoder(self.config.encoder)
        self.transformer    = SpectralTransformer(self.config.transformer)
        self.physics_head   = PhysicsHead(
            num_groups=self.config.encoder.num_groups_out
        )
        self.beer_lambert   = BeerLambertConstraint(
            implausibility_threshold=self.config.implausibility_threshold
        )
        self.group_freq_penalty = GroupFrequencyPenalty()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        spectrum: torch.Tensor,               # (B, W)
        tgt_tokens: torch.Tensor,              # (B, T)
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns dict with keys:
            logits        : (B, T, V)  SMILES token logits
            reconstructed : (B, W)     physics-reconstructed spectrum
            group_logits  : (B, G)     group presence logits
            bl_loss       : scalar     Beer–Lambert constraint loss
            gf_loss       : scalar     group frequency penalty
            bl_report     : BeerLambertResidual diagnostics
        """
        # 1. Encode spectrum
        spectral_tokens, group_logits = self.encoder(spectrum)

        # 2. Physics head: reconstruct spectrum from group predictions
        group_probs, reconstructed_spectrum = self.physics_head(group_logits)

        # 3. Beer–Lambert constraint
        bl_loss, bl_report = self.beer_lambert(spectrum, reconstructed_spectrum)

        # 4. Group frequency penalty
        gf_loss = self.group_freq_penalty(spectrum, group_logits)

        # 5. Autoregressive SMILES decoding
        logits = self.transformer(spectral_tokens, tgt_tokens, tgt_key_padding_mask)

        return {
            "logits":        logits,
            "reconstructed": reconstructed_spectrum,
            "group_logits":  group_logits,
            "bl_loss":       bl_loss,
            "gf_loss":       gf_loss,
            "bl_report":     bl_report,
        }

    @torch.no_grad()
    def predict(
        self,
        spectrum: torch.Tensor,       # (B, W)
        beam_size: int = 4,
        return_diagnostics: bool = True,
    ) -> dict:
        """
        Inference: spectrum → SMILES + physics diagnostics.

        Returns dict with keys:
            smiles_tokens       : (B, T) generated token sequences
            log_probs           : (B,)   sequence log-probabilities
            reconstructed_spec  : (B, W) Lorentzian-reconstructed spectrum
            group_probs         : (B, G) per-group confidence
            ecr                 : (B,)   Energy Conservation Residual
            implausible         : (B,)   bool ECR > threshold
            implausible_rate    : float  fraction implausible
        """
        self.eval()
        spectral_tokens, group_logits = self.encoder(spectrum)
        group_probs, reconstructed    = self.physics_head(group_logits)
        _, bl_report = self.beer_lambert(spectrum, reconstructed)

        smiles_tokens, log_probs = self.transformer.generate(
            spectral_tokens, beam_size=beam_size
        )

        return {
            "smiles_tokens":     smiles_tokens,
            "log_probs":         log_probs,
            "reconstructed_spec": reconstructed,
            "group_probs":       group_probs,
            "ecr":               bl_report.ecr,
            "implausible":       bl_report.physically_implausible,
            "implausible_rate":  bl_report.implausible_rate,
        }