"""
Cross-attention Transformer: spectral tokens → SMILES tokens.

The encoder-decoder architecture treats the spectral token sequence as
"source" and the SMILES/IUPAC sequence as "target". Cross-attention allows
each output token to attend over the entire spectral sequence — learning
which spectral regions are informative for each part of the molecular name.

This is semantically analogous to machine translation, but the "source language"
is physics and the "target language" is chemistry nomenclature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ── SMILES vocabulary ─────────────────────────────────────────────────────────
# A minimal but complete SMILES tokeniser vocabulary.
# Note: we tokenise at the character/symbol level, not subword — this preserves
# the syntactic structure of SMILES strings (ring closures, branches, etc.)
SMILES_VOCAB = [
    "<pad>", "<bos>", "<eos>", "<unk>",  # special tokens
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",  # atoms
    "c", "n", "o", "s",  # aromatic atoms
    "1", "2", "3", "4", "5", "6", "7", "8", "9",  # ring closures
    "(", ")", "[", "]",  # branches and formal charge notation
    "=", "#", "-", "+",  # bond types and charges
    "/", "\\",           # stereochemistry
    "H", "@", ".",       # hydrogen, chirality, disconnected
    "%10", "%11", "%12",  # extended ring closures
]
VOCAB_SIZE = len(SMILES_VOCAB)
PAD_IDX = SMILES_VOCAB.index("<pad>")
BOS_IDX = SMILES_VOCAB.index("<bos>")
EOS_IDX = SMILES_VOCAB.index("<eos>")


def smiles_tokenise(smiles: str) -> list[int]:
    """Character-level SMILES tokeniser. Returns token index list."""
    tokens = [BOS_IDX]
    i = 0
    while i < len(smiles):
        # Two-character tokens first (Cl, Br, %10–%12)
        if smiles[i:i+2] in SMILES_VOCAB:
            tokens.append(SMILES_VOCAB.index(smiles[i:i+2]))
            i += 2
        elif smiles[i] in SMILES_VOCAB:
            tokens.append(SMILES_VOCAB.index(smiles[i]))
            i += 1
        else:
            tokens.append(SMILES_VOCAB.index("<unk>"))
            i += 1
    tokens.append(EOS_IDX)
    return tokens


def smiles_detokenise(tokens: list[int]) -> str:
    """Convert token indices back to SMILES string."""
    chars = []
    for t in tokens:
        if t in (PAD_IDX, BOS_IDX):
            continue
        if t == EOS_IDX:
            break
        if 0 <= t < VOCAB_SIZE:
            chars.append(SMILES_VOCAB[t])
    return "".join(chars)


@dataclass
class TransformerConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4    # Transformer encoder (over spectral tokens)
    num_decoder_layers: int = 4    # Transformer decoder (SMILES generation)
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_smiles_len: int = 128      # Max SMILES output length
    vocab_size: int = VOCAB_SIZE
    pad_idx: int = PAD_IDX


class SpectralTransformer(nn.Module):
    """
    Full encoder-decoder Transformer for spectrum → SMILES translation.

    The encoder refines the spectral token sequence (from SpectralEncoder)
    with self-attention — learning correlations between spectral regions
    (e.g., that a carbonyl peak at 1710 cm⁻¹ and an O-H stretch at 3300 cm⁻¹
    co-occur in carboxylic acids).

    The decoder generates SMILES autoregressively via teacher forcing during
    training and greedy / beam search during inference.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # SMILES token embedding
        self.tgt_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        # Learned positional embedding for SMILES (not physical, so learned is fine)
        self.tgt_pos_embedding = nn.Embedding(config.max_smiles_len, config.d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-norm for training stability
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, spectral_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral_tokens: (B, W', d_model) from SpectralEncoder
        Returns:
            memory: (B, W', d_model) — contextualised spectral memory
        """
        return self.encoder(spectral_tokens)

    def decode(
        self,
        tgt_tokens: torch.Tensor,   # (B, T) — SMILES token indices
        memory: torch.Tensor,        # (B, W', d_model) — spectral memory
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        B, T = tgt_tokens.shape
        positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)  # (1, T)

        tgt_emb = (
            self.tgt_embedding(tgt_tokens)
            + self.tgt_pos_embedding(positions)
        )  # (B, T, d_model)

        # Causal mask: token i can only attend to tokens 0..i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device
        )  # (T, T)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, d_model)

        return self.output_proj(out)  # (B, T, vocab_size)

    def forward(
        self,
        spectral_tokens: torch.Tensor,   # (B, W', d_model)
        tgt_tokens: torch.Tensor,         # (B, T) — shifted right for teacher forcing
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass. Returns logits (B, T, vocab_size)."""
        memory = self.encode(spectral_tokens)
        return self.decode(tgt_tokens, memory, tgt_key_padding_mask)

    @torch.no_grad()
    def generate(
        self,
        spectral_tokens: torch.Tensor,   # (B, W', d_model)
        max_len: int | None = None,
        temperature: float = 1.0,
        beam_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy or beam search autoregressive SMILES generation.

        Returns:
            token_ids : (B, T) generated token sequences
            log_probs : (B,)   sequence log-probabilities
        """
        max_len = max_len or self.config.max_smiles_len
        memory = self.encode(spectral_tokens)
        B = spectral_tokens.size(0)
        device = spectral_tokens.device

        if beam_size == 1:
            return self._greedy_decode(memory, B, max_len, temperature, device)
        else:
            return self._beam_decode(memory, B, max_len, beam_size, device)

    def _greedy_decode(
        self,
        memory: torch.Tensor,
        B: int,
        max_len: int,
        temperature: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy autoregressive decoding."""
        generated = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        log_probs = torch.zeros(B, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decode(generated, memory)  # (B, T, V)
            next_logits = logits[:, -1, :] / temperature  # (B, V)
            next_log_p = F.log_softmax(next_logits, dim=-1)  # (B, V)
            next_token = next_log_p.argmax(dim=-1)  # (B,)

            # Accumulate log-probs for non-finished sequences
            log_probs += next_log_p.gather(1, next_token.unsqueeze(1)).squeeze(1) * ~finished

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == EOS_IDX)

            if finished.all():
                break

        return generated, log_probs

    def _beam_decode(
        self,
        memory: torch.Tensor,
        B: int,
        max_len: int,
        beam_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding. Handles batch_size=1 for now."""
        # For simplicity: expand memory for beam
        assert B == 1, "Beam search currently supports batch_size=1"
        memory_exp = memory.expand(beam_size, -1, -1)  # (beam, W', d_model)

        # Beams: list of (log_prob, token_sequence)
        beams: list[tuple[float, list[int]]] = [(0.0, [BOS_IDX])]
        completed: list[tuple[float, list[int]]] = []

        for step in range(max_len - 1):
            candidates = []
            for log_p, tokens in beams:
                if tokens[-1] == EOS_IDX:
                    completed.append((log_p, tokens))
                    continue
                tgt = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)
                logits = self.decode(tgt, memory)  # (1, T, V)
                next_log_p = F.log_softmax(logits[0, -1, :], dim=-1)  # (V,)
                top_logp, top_tokens = next_log_p.topk(beam_size)

                for lp, tok in zip(top_logp.tolist(), top_tokens.tolist()):
                    candidates.append((log_p + lp, tokens + [tok]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            if len(completed) >= beam_size:
                break

        completed.extend(beams)
        completed.sort(key=lambda x: x[0], reverse=True)
        best_lp, best_tokens = completed[0]

        output = torch.tensor([best_tokens], dtype=torch.long, device=device)
        log_prob = torch.tensor([best_lp], device=device)
        return output, log_prob