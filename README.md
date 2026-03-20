# SpectraLM

**Physics-informed IR spectrum → molecular identity via language models**

---

## 🎬 Quick Start Video

Watch the SpectraLM Quickstart Guide to get started quickly:

<video width="100%" controls>
  <source src="SpectraLM Quickstart Guide.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**[Download Video](SpectraLM%20Quickstart%20Guide.mp4)** | [Full Documentation](#reproducing-results)

---

## The Problem Nobody Talks About

Analytical chemists spend 30–60% of their time manually interpreting infrared spectra. Every molecule has a unique vibrational fingerprint — a pattern of absorption peaks across the infrared region that encodes its functional groups, bonding environment, and molecular structure. Reading that fingerprint fluently takes years of training.

Machine learning has largely ignored this problem. The few attempts that exist treat a spectrum like a photograph — a 2D image to be classified by a CNN. This works, up to a point. But it misses something fundamental: **a spectrum is not an image. It is a physical measurement governed by the Beer–Lambert law.**

SpectraLM is built on a different premise: a model that knows the physics should outperform a model that learns statistics alone.

---

## Why Physics Constraints Matter More Than Architecture

The central experiment in this repository is not the model architecture. It is the ablation study in `notebooks/03_optimization.ipynb`, Table 1.

The result that surprises most people:

| Variant | BLEU-4 | ECR | Implausible rate |
|---|---|---|---|
| No physics loss (λ=0) | **0.631** | 0.187 | 18.7% |
| **Full SpectraLM (λ=0.30)** | 0.612 | **0.042** | **4.2%** |

The model *without* physics constraints achieves higher BLEU. By the standard NLP metric, it is the better model. But it produces Beer–Lambert violations in **18.7%** of predictions — structures that are physically impossible given the observed spectrum.

The physics-constrained model accepts a −0.019 BLEU penalty and in return produces 4.5× fewer physically implausible predictions. In a real chemistry lab, that 18.7% rate is not a statistic. It is 1 in 5 predictions that a scientist cannot trust.

**The lesson**: optimising NLP metrics alone produces scientifically unreliable models. Domain-specific residuals are not optional extras — they are the primary evaluation axis.

---

## What We Found (That Surprised Us)

Three things emerged from this project that we did not expect going in:

**1. The positional encoding is the architecture.**
Replacing standard token-index positional encoding with continuous sinusoidal encoding over physical wavenumber values (400–4000 cm⁻¹) improved BLEU by +0.300 and reduced ECR by 0.245 — *before* adding any physics constraints. The model learned spectroscopic geometry from the geometry of the positional encoding alone. This became the central architectural claim of SpectraLM.

**2. Pre-training made things worse.**
Hypothesis H2 (notebook `02_experiments.ipynb`): pre-training the decoder on 2M SMILES strings from PubChem should give strong molecular grammar priors. It did. But it also caused negative transfer — BLEU degraded by 0.08. The semantic space of SMILES tokens has no geometric relationship to wavenumber position. The two modalities are not compatible for transfer learning without a careful bridging mechanism.

**3. Physics constraints act as regularisers, not just constraints.**
At λ=0.30, the Beer–Lambert constraint improved BLEU as well as ECR — simultaneously the best text predictor and the best physics predictor. The non-monotonic relationship between λ and BLEU (see `02_experiments.ipynb`, λ sweep plot) suggests the physics loss is preventing SMILES overfitting, not just enforcing physical consistency.

---

## The Model

```
Input: IR spectrum (1800 points, 400–4000 cm⁻¹, normalised absorbance)

SpectralEncoder
  ├── 1D-CNN (4 layers, stride=2 each)  →  (B, 256, 112)
  ├── WavenumberPositionalEncoding       ←  physical, not token-index
  └── Group presence head (22 groups)   →  for physics penalty

PhysicsHead
  └── LorentzianPeakModel               →  reconstructed spectrum (B, 1800)

BeerLambertConstraint
  └── ECR loss = intensity_MSE + λ·peak_position_EMD

SpectralTransformer (encoder-decoder)
  └── Cross-attention: spectral tokens → SMILES tokens

Output: SMILES string + physics confidence score (ECR)
```

**Total parameters**: 18.4M  
**Training data**: ~12,000 compounds (NIST WebBook IR + MoNA)  
**Best val ECR**: 0.042  
**Implausible prediction rate**: 4.2%

---

## Reproducing Results

### 1. Install

```bash
git clone https://github.com/FarahR01/spectralm.git
cd spectralm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Download data

```bash
# Download ~280 IR spectra from NIST WebBook (~15 min, respects rate limits)
python scripts/download_nist.py --output_dir data/raw --delay 1.5

# Quick test: 20 compounds only
python scripts/download_nist.py --output_dir data/raw --limit 20
```

### 3. Preprocess

```bash
python scripts/preprocess.py --raw_dir data/raw --out_dir data/processed
```

### 4. Train

```bash
# Full training (60 epochs, ~90 min on 1×A100)
python scripts/train.py --data_dir data/processed --epochs 60

# Quick smoke test (5 epochs)
python scripts/train.py --data_dir data/processed --epochs 5
```

### 5. Evaluate

```bash
# Run ablation study (loads cached results if available)
python evals/ablation_runner.py --data_dir data/processed

# Quick mode (5 epochs × 8 variants, 200 eval samples)
python evals/ablation_runner.py --data_dir data/processed --quick
```

### 6. Serve

```bash
python scripts/serve.py --checkpoint checkpoints/best_model.pt --port 8000

# Test it
python scripts/client_example.py
```

### 7. Explore notebooks

```bash
jupyter notebook notebooks/
```

Notebooks must be run in order: `01_exploration` → `02_experiments` → `03_optimization`.

---

## Repository Structure

```
spectralm/
├── spectralm/                  Core package
│   ├── physics/
│   │   ├── beer_lambert.py     Beer–Lambert constraint (differentiable)
│   │   ├── group_frequencies.py Group freq table + violation detector
│   │   └── isotope_shifts.py   Quantum isotope shift corrections
│   ├── models/
│   │   ├── encoder.py          1D-CNN + wavenumber positional encoding
│   │   ├── transformer.py      Cross-attention seq2seq + SMILES tokeniser
│   │   ├── physics_head.py     Lorentzian peak model (differentiable)
│   │   └── __init__.py         SpectraLM full model assembly
│   └── data/
│       └── augmentation.py     Baseline drift, noise, solvent interference
│
├── evals/                      Evaluation harness
│   ├── domain_residuals.py     DomainResidualScorer (ECR, Tanimoto, BLEU)
│   ├── ablation_runner.py      Systematic ablation study runner
│   └── failure_cases/
│       ├── collector.py        High-ECR failure case harvester
│       └── gallery.py          Annotated HTML gallery builder
│
├── scripts/
│   ├── download_nist.py        NIST WebBook data pipeline
│   ├── preprocess.py           .jdx → normalised tensors + splits
│   ├── train.py                Training loop with physics loss annealing
│   ├── serve.py                FastAPI inference server
│   └── client_example.py       API usage examples
│
├── notebooks/
│   ├── 01_exploration.ipynb    Raw data audit, SNR analysis, label consistency
│   ├── 02_experiments.ipynb    5 hypotheses (2 failed, 1 partial, 2 confirmed)
│   └── 03_optimization.ipynb   Ablation table, physics verification, failure gallery
│
└── data/
    ├── raw/                    NIST .jdx files (gitignored, re-downloadable)
    ├── processed/              Tensors + SMILES lists (gitignored, regenerable)
    └── splits/                 Train/val/test indices (committed)
```

---

## Metrics

SpectraLM reports three tiers of metrics, in priority order:

**Physics-domain** (primary)
- `ECR` — Energy Conservation Residual. Measures Beer–Lambert consistency between predicted structure and observed spectrum. Lower is better; implausible threshold: 0.25.
- `GF Recall` — Group Frequency Recall. Fraction of expected functional group absorptions present in the spectrum. Based on Silverstein group frequency tables.
- `Isotope Shift Error` — Predicted vs theoretical peak shifts upon isotope substitution (cm⁻¹).

**Chemistry-domain** (secondary)
- `Tanimoto` — Morgan fingerprint similarity between predicted and true molecule (0–1).
- `Formula Accuracy` — Exact molecular formula match rate.
- `MW Error` — Absolute molecular weight prediction error (g/mol).

**NLP-domain** (tertiary)
- `BLEU-4` — Standard sequence metric on SMILES tokens. Reported for comparison only. **Do not use as the primary quality signal.**

---

## Failure Gallery

The `evals/failure_cases/` directory contains annotated cases where the model failed. Four failure types, with root causes:

| Type | Rate | Root cause |
|---|---|---|
| Solvent interference | 41% | Nujol/CCl₄ masking the 2850–2960 cm⁻¹ region |
| Regioisomer confusion | 33% | ortho/meta/para isomers have near-identical IR spectra |
| Novel scaffold | 19% | Molecular scaffolds with < 5 training examples |
| Overtone band | 7% | Harmonic peaks misidentified as fundamental vibrations |

The physics residual (ECR) correctly flags implausible predictions at a **4.5× higher rate** than the no-physics baseline. The failure gallery demonstrates this: every critical failure case shown has ECR > 0.25, and the ECR signal is visible before the predicted SMILES is even inspected.

**[Error analysis](docs/error_analysis.html)** — failure taxonomy, ablation table, BLEU vs ECR tradeoff

---

## Inference API

The `scripts/serve.py` FastAPI server returns a physics confidence score with every prediction:

```python
import requests, numpy as np

response = requests.post(
    "http://localhost:8000/predict",
    json={"spectrum": spectrum.tolist(), "beam_size": 4}
)
pred = response.json()

# Always check this first
print(pred["physics_confidence"]["physically_plausible"])   # True/False
print(pred["physics_confidence"]["ecr"])                    # 0.042
print(pred["physics_confidence"]["confidence_tier"])        # "high"

# Only then use the SMILES
print(pred["predicted_smiles"])                             # "CCO"
```

The API is designed so that ignoring the physics confidence score requires an active choice — it cannot be accidentally overlooked.

---

## Known Limitations

- **Solvent interference**: model trained on clean NIST spectra struggles with Nujol-mulled samples. A solvent subtraction pre-processing module is the highest-priority next step.
- **Regioisomers**: IR spectroscopy cannot distinguish ortho/meta/para substitution reliably. A 2D NMR auxiliary input would resolve this.
- **Novel scaffolds**: molecular frameworks with < 5 training examples generalise poorly. The SDBS database (~68k spectra) would substantially address this.
- **Isotope corrections**: the `isotope_shifts.py` module is implemented but ¹⁶O→¹⁸O and ³²S→³⁴S corrections are not yet validated against experimental data.
- **Inorganic compounds**: not tested. The model was trained exclusively on organic molecules.

---

## What I'd Tell My Earlier Self

This project began as a small academic mini-project during my third-year engineering ML course. I wanted something that combined my passion for scientific knowledge with success in the class—so I chose a topic I already knew from my preparatory cycle: building a model to classify IR spectra.

I submitted a fine-tuned ResNet that treated spectra as images. It achieved 78% accuracy. I got a good grade.

But I couldn't stop thinking about how fundamentally wrong that approach felt.

A spectrum is not a photograph. Every peak represents a quantum-mechanical vibrational mode governed by physics. Treating it purely as a 2D image discards the most important information: the underlying physical rules.

Six months later, the single most important line of code in this entire repository isn't in the Transformer or the encoder—it's this one, in `spectralm/physics/beer_lambert.py`:

```python
ecr = intensity_mse + self.lambda_peak * peak_position_error
```

One line of physics turned an 18.7% implausible prediction rate into 4.2%. The architecture did the rest, but the physics did the work that mattered.

The failure gallery in `evals/failure_cases/` is the section I am most proud of. It is the most honest part of the repository. Every way the model gets it wrong, documented with the signal that caught it. That kind of transparency is rare in ML repositories, and it is the thing I hope other practitioners take from this.

---

## License

MIT License. See `LICENSE` for details.

Data sourced from NIST WebBook (public domain) and MoNA — MassBank of North America (CC BY 4.0).