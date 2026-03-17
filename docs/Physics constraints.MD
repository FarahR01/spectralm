# Physics Constraints in SpectraLM

## Why Constrain at All?

A standard seq2seq model trained to predict SMILES from spectra will learn whatever statistical patterns minimise cross-entropy loss. If the training data says "spectrum X → ethanol", the model learns that association. This works surprisingly well.

But it has a silent failure mode: the model can generate a *grammatically valid* SMILES string that is *physically inconsistent* with the observed spectrum. It produces a molecule whose IR fingerprint — if you were to compute it from first principles — would look nothing like what was measured.

In a lab, this would be caught immediately. A chemist would look at the proposed structure, compute its expected absorptions mentally, compare them to the spectrum, and say: *"That can't be right — where's the carbonyl peak?"*

SpectraLM's physics constraints do this automatically, during training, as a differentiable loss term.

---

## Constraint 1 — Beer–Lambert Energy Conservation Residual (ECR)

**What it is:**
The primary physics loss. Measures whether the spectrum *reconstructed from the model's predicted molecular structure* is consistent with the *observed input spectrum*.

**The physics:**
Beer–Lambert law states that absorbance is linear in concentration and path length:
```
A(ν) = ε(ν) · c · l
```
For our purposes: if we know what the molecule is (or think we do), we can *predict* what its spectrum should look like. If the prediction disagrees strongly with what we actually measured — that's a physics violation.

**How it's computed:**
```
ECR = intensity_MSE + λ_peak · peak_position_EMD

intensity_MSE    = (1/W) · ||A_observed − A_reconstructed||²
peak_position_EMD = earth-mover distance between observed
                    and reconstructed peak distributions
```

The Earth-Mover Distance (EMD) component penalises predictions where peaks appear at the *wrong wavenumber positions* — not just wrong intensities. This matters because a model might reconstruct the right peak heights but assign them to the wrong functional groups.

**How it's used in training:**
```python
bl_loss, bl_report = self.beer_lambert(observed_spectrum, reconstructed_spectrum)
total_loss = ce_loss + 0.30 * bl_loss + 0.15 * gf_loss
```

Lambda = 0.30 was determined by grid search (see `notebooks/02_experiments.ipynb`, λ sweep). This value gives the best BLEU *and* the best ECR simultaneously — suggesting the constraint acts as a physics-informed regulariser, not just a penalty.

**At inference:**
Every prediction from `serve.py` includes:
```json
"physics_confidence": {
  "ecr": 0.042,
  "physically_plausible": true,
  "confidence_tier": "high"
}
```
ECR > 0.25 → prediction flagged as physically implausible. Do not use in a lab without further verification.

**Performance:**
| Model variant | ECR (mean) | Implausible rate |
|---|---|---|
| No physics loss | 0.187 | 18.7% |
| Full SpectraLM | **0.042** | **4.2%** |

---

## Constraint 2 — Group Frequency Penalty (GF Penalty)

**What it is:**
A differentiable penalty that fires when the model predicts a functional group is present in the molecule but the corresponding spectral region shows no absorption.

**The physics:**
Every functional group absorbs at a characteristic, reproducible wavenumber range. This is Silverstein's group frequency principle — the empirical foundation of manual IR interpretation. If a model predicts a ketone but there is no absorption in the 1680–1750 cm⁻¹ C=O stretch region, something is wrong.

**How it's computed:**
For each of 22 functional groups, the penalty checks:
```
For each group g with predicted probability p_g:
    Find the maximum absorbance in group g's expected wavenumber range
    If max_absorbance < required_minimum:
        penalty += p_g · (required_minimum − max_absorbance)
```

The penalty is weighted by the model's confidence in the group prediction — so a group predicted with 90% confidence but missing from the spectrum contributes more to the loss than one predicted with 10% confidence.

**Why it complements the BL constraint:**
The BL constraint operates on the full spectrum shape. The GF penalty operates on specific, named spectral regions tied to chemical knowledge. Together they enforce both global spectral consistency (BL) and local functional group assignments (GF).

**Performance:**
Removing the GF penalty in ablation (Table 1, row 4):
- BLEU change: +0.001 (negligible)
- ECR change: +0.114 (significant worsening)

Pure physics gain at near-zero text-prediction cost.

---

## Constraint 3 — Lorentzian Peak Reconstruction (The Forward Model)

**What it is:**
Not a loss term directly, but the mechanism that makes constraints 1 and 2 differentiable. Given the model's predicted group presence probabilities, this module reconstructs what the IR spectrum *should* look like.

**The physics:**
IR absorption bands are Lorentzian-shaped due to homogeneous lifetime broadening from molecular collisions:
```
A(ν) = I_max / (1 + ((ν − ν₀) / (Δν/2))²)
```

Where `ν₀` is learnable (constrained to each group's physical range) and `Δν` (FWHM) is also learnable. This makes the reconstruction differentiable — gradients flow from the BL residual back through the Lorentzian shapes, through the group probabilities, all the way to the encoder.

**Why Lorentzian and not Gaussian?**
Gaussian broadening is *inhomogeneous* — it models a distribution of slightly different environments. Lorentzian broadening is *homogeneous* — it models lifetime effects. In gas-phase IR, Lorentzian dominates. In condensed-phase (which is what NIST WebBook spectra are), the bands are typically Voigt profiles (convolution of Gaussian and Lorentzian), but Lorentzian is a good approximation and cheaper to compute.

**Architecture position:**
```
encoder → group_logits (B, 22)
               ↓
         LorentzianPeakModel
               ↓
    reconstructed_spectrum (B, 1800)
               ↓
    BeerLambertConstraint → ECR loss
```

---

## Constraint 4 — Isotope Shift Correction (Partial)

**What it is:**
A soft regulariser based on quantum harmonic oscillator theory. When a molecule contains a heavy isotope (¹³C instead of ¹²C, D instead of H), its vibrational frequencies shift predictably according to the reduced mass ratio.

**The physics:**
For a diatomic approximation:
```
ν_heavy / ν_light = √(μ_light / μ_heavy)

where μ = (m₁ · m₂) / (m₁ + m₂)  [reduced mass]
```

The force constant `k` is unchanged by isotope substitution — it depends on electronic structure, not nuclear mass. So the frequency shift is determined purely by the mass change.

**Example — H → D (deuteration):**
```
μ(C-H) = (12 × 1.008) / (12 + 1.008) = 0.929 u
μ(C-D) = (12 × 2.014) / (12 + 2.014) = 1.722 u

ν(C-D) / ν(C-H) = √(0.929 / 1.722) = 0.735

C-H stretch at 2920 cm⁻¹ → C-D stretch at ~2145 cm⁻¹
Δν ≈ −775 cm⁻¹
```

**Status in SpectraLM:**
H→D and ¹²C→¹³C corrections are implemented and validated. ¹⁶O→¹⁸O and ³²S→³⁴S are implemented but not yet validated against experimental data. The `IsotopeShiftPenalty` module exists but is not yet included in the default training loss (marked in model card as "next steps").

---

## Physics Annealing — Why the Constraints Are Introduced Gradually

A critical implementation detail: the physics losses are not applied at full weight from epoch 0. They are linearly annealed from 0 → full weight over the first 10 epochs.

**Why:**
Early in training, the SMILES decoder hasn't learned basic molecular grammar yet — ring closure notation, valid valence, branch syntax. If the physics constraint fires at full strength from epoch 0, it overwhelms the CE loss and the model collapses to predicting only the simplest molecules (those with the easiest-to-reconstruct spectra).

By waiting 10 epochs for the decoder to stabilise, the physics constraint finds a model that already knows how to generate valid SMILES, and steers it toward physical consistency rather than fighting basic syntax learning.

```python
def physics_lambda_schedule(epoch: int, warmup_epochs: int = 10) -> float:
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs
```

This was discovered empirically — without annealing, training collapsed in 3 of 5 random seeds. With annealing, training was stable in all 5.

---

## How the Constraints Interact

```
              ┌─────────────────────────────────┐
              │         Training loss            │
              │                                  │
              │  L = CE_loss                     │
              │    + 0.30 · BL_ECR               │  ← enforces global
              │    + 0.15 · GF_penalty           │  ← enforces local
              └─────────────────────────────────┘
                         ↑
              ┌──────────┴──────────┐
              │                     │
       BL_ECR computed via    GF_penalty computed via
       LorentzianPeakModel    GroupFrequencyPenalty
       (differentiable)       (differentiable)
              │
       Requires: spectrum reconstruction
       Which requires: group probability predictions
       Which requires: encoder output
              │
       → Gradients flow all the way back to the CNN encoder
       → The encoder learns to extract group-informative features
         not just reconstruction-informative features
```

The key insight is that the physics constraints are not post-hoc filters — they are in the gradient path. The encoder is trained to extract features that are useful for both SMILES generation *and* physical spectrum reconstruction simultaneously.

---

## Summary Table

| Constraint | Module | λ | Ablation impact | Status |
|---|---|---|---|---|
| Beer–Lambert ECR | `beer_lambert.py` | 0.30 | ECR: −0.145 if removed | ✓ Production |
| Group Frequency Penalty | `group_frequencies.py` | 0.15 | ECR: −0.114 if removed | ✓ Production |
| Lorentzian Reconstruction | `physics_head.py` | — | Enables BL constraint | ✓ Production |
| Isotope Shift Penalty | `isotope_shifts.py` | 0.05 | Not yet ablated | ~ Partial |