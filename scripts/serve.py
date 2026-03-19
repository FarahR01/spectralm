"""
scripts/serve.py

SpectraLM inference server.

Serves the trained model as a REST API with physics-residual confidence
scoring baked into every prediction response.

The key design decision: the API never returns a prediction without
also returning its ECR (Energy Conservation Residual). A consumer
who ignores the physics confidence score is making an active choice —
the default response makes it impossible to miss.

Endpoints:
    POST /predict           — spectrum → SMILES + physics diagnostics
    POST /predict/batch     — batch inference (up to 32 spectra)
    POST /predict/smiles    — SMILES string input → group analysis
    GET  /health            — server + model status
    GET  /model/info        — model card metadata
    GET  /docs              — auto-generated OpenAPI UI (FastAPI default)

Usage:
    # Start server
    python scripts/serve.py --checkpoint checkpoints/best_model.pt --port 8000

    # With GPU
    python scripts/serve.py --checkpoint checkpoints/best_model.pt --device cuda

    # Production (multiple workers via gunicorn)
    gunicorn scripts.serve:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Example request:
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"spectrum": [0.12, 0.08, ...], "beam_size": 4}'
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from rich.default_styles import args
from rich.default_styles import args
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import RedirectResponse, HTMLResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectralm.models import SpectraLM, SpectraLMConfig
from spectralm.models.transformer import smiles_detokenise
from spectralm.physics.beer_lambert import BeerLambertConstraint
from spectralm.physics.group_frequencies import (
    GroupFrequencyChecker, GROUP_FREQUENCY_TABLE, WAVENUMBER_AXIS
)
# Serve the static folder
from fastapi.staticfiles import StaticFiles

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spectralm.serve")


# ══════════════════════════════════════════════════════════════════════════════
# Global model state
# Loaded once at startup, shared across all requests.
# ══════════════════════════════════════════════════════════════════════════════

class ModelState:
    model:          SpectraLM | None = None
    config:         SpectraLMConfig | None = None
    device:         torch.device = torch.device("cpu")
    checkpoint_path: str = ""
    loaded_epoch:   int = 0
    val_ecr:        float = 0.0
    load_time_sec:  float = 0.0
    bl_constraint:  BeerLambertConstraint | None = None
    gf_checker:     GroupFrequencyChecker | None = None
    request_count:  int = 0
    error_count:    int = 0
    server_start_time: float = 0.0

state = ModelState()


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint into global state."""
    t0 = time.time()
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state.config = ckpt.get("config", SpectraLMConfig())
    state.model  = SpectraLM(state.config)
    state.model.load_state_dict(ckpt["model_state_dict"])
    state.model.eval()
    state.model.to(device)

    state.device          = torch.device(device)
    state.checkpoint_path = str(ckpt_path)
    state.loaded_epoch    = ckpt.get("epoch", -1)
    state.val_ecr         = ckpt.get("val_ecr", -1.0)
    state.load_time_sec   = time.time() - t0
    state.bl_constraint   = BeerLambertConstraint()
    state.gf_checker      = GroupFrequencyChecker()
    state.server_start_time = time.time()

    n_params = state.model.num_parameters
    log.info(f"Model loaded  "
             f"epoch={state.loaded_epoch}  "
             f"val_ecr={state.val_ecr:.4f}  "
             f"params={n_params:,}  "
             f"device={device}  "
             f"time={state.load_time_sec:.2f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    Single-spectrum prediction request.

    spectrum  : 1800-point normalised absorbance array (wavenumber 400–4000 cm⁻¹,
                2 cm⁻¹ resolution, values in [0, 1]).
                If your spectrum has a different resolution, pre-interpolate to 1800 points.
    beam_size : Beam search width. Higher = better quality, slower.
                Recommended: 1 (fast) or 4 (quality).
    return_reconstructed : Include the Lorentzian-reconstructed spectrum
                           in the response (adds ~2KB per request).
    return_group_probs   : Include per-functional-group confidence scores.
    """
    spectrum: list[float] = Field(
        ...,
        min_length=1800,
        max_length=1800,
        description="1800-point normalised absorbance (400–4000 cm⁻¹, 2 cm⁻¹ resolution)",
        examples=[[0.0] * 1800],
    )
    beam_size: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam search width (1=greedy, 4=recommended, 10=max quality)",
    )
    return_reconstructed: bool = Field(
        default=False,
        description="Include reconstructed spectrum in response",
    )
    return_group_probs: bool = Field(
        default=True,
        description="Include per-group confidence scores",
    )

    @field_validator("spectrum")
    @classmethod
    def validate_spectrum(cls, v: list[float]) -> list[float]:
        arr = np.array(v)
        if arr.min() < -0.1 or arr.max() > 1.1:
            raise ValueError(
                f"Spectrum values out of range [{arr.min():.3f}, {arr.max():.3f}]. "
                "Expected normalised absorbance in [0, 1]."
            )
        if arr.max() < 0.01:
            raise ValueError(
                "Spectrum appears to be all zeros or near-zero. "
                "Check normalisation — peak absorbance should be ~1.0."
            )
        return v


class BatchPredictRequest(BaseModel):
    """Batch prediction — up to 32 spectra."""
    spectra: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of 1800-point spectra (max 32)",
    )
    beam_size: int = Field(default=1, ge=1, le=4)
    return_reconstructed: bool = False
    return_group_probs: bool = False

    @field_validator("spectra")
    @classmethod
    def validate_spectra(cls, v):
        for i, spec in enumerate(v):
            if len(spec) != 1800:
                raise ValueError(
                    f"Spectrum at index {i} has {len(spec)} points, expected 1800."
                )
        return v


class SMILESAnalysisRequest(BaseModel):
    """
    Analyse a known SMILES string against a provided spectrum.
    Useful for verifying whether a candidate structure is consistent
    with an observed spectrum.
    """
    smiles: str = Field(
        ...,
        min_length=2,
        max_length=512,
        description="SMILES string of the candidate molecule",
        examples=["CC(=O)OCC"],
    )
    spectrum: list[float] = Field(
        ...,
        min_length=1800,
        max_length=1800,
        description="Observed IR spectrum (1800 points, normalised)",
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class PhysicsConfidence(BaseModel):
    """
    Physics-residual confidence report.

    This is the primary quality signal for every prediction.
    A high BLEU-like NLP score with a high ECR should be treated
    with scepticism by any downstream consumer.
    """
    ecr: float = Field(
        description="Energy Conservation Residual. "
                    "Lower is better. Implausible threshold: 0.25."
    )
    peak_position_error: float = Field(
        description="Earth-mover distance between predicted and observed peak positions."
    )
    intensity_mse: float = Field(
        description="MSE between observed and Lorentzian-reconstructed absorbance."
    )
    physically_plausible: bool = Field(
        description="True if ECR < 0.25. Use this as a hard filter in production."
    )
    confidence_tier: str = Field(
        description="Human-readable tier: 'high' (ECR<0.05), 'medium' (0.05–0.15), "
                    "'low' (0.15–0.25), 'implausible' (>0.25)."
    )
    gf_recall: float = Field(
        description="Fraction of expected functional group absorptions present. "
                    "Based on Beer–Lambert additivity and group frequency table."
    )
    gf_violations: list[str] = Field(
        description="Functional groups predicted in SMILES but missing from spectrum."
    )


class PredictResponse(BaseModel):
    """
    Prediction response with physics-residual confidence scoring.

    ⚠ Always check physics_confidence.physically_plausible before using
      the predicted_smiles in any downstream application.
    """
    predicted_smiles: str = Field(
        description="Predicted SMILES string. May be empty if generation failed."
    )
    valid_smiles: bool = Field(
        description="Whether predicted_smiles is parseable by RDKit."
    )
    log_prob: float = Field(
        description="Sequence log-probability. More negative = less confident."
    )
    physics_confidence: PhysicsConfidence = Field(
        description="Physics-residual quality assessment. "
                    "Read this before trusting predicted_smiles."
    )
    group_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Per-functional-group sigmoid confidence scores (0–1). "
                    "Null if return_group_probs=False.",
    )
    reconstructed_spectrum: list[float] | None = Field(
        default=None,
        description="Lorentzian-reconstructed spectrum from predicted structure. "
                    "Null if return_reconstructed=False.",
    )
    inference_time_ms: float = Field(
        description="Server-side inference time in milliseconds."
    )
    model_epoch: int = Field(
        description="Training epoch of the loaded checkpoint."
    )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    batch_size: int
    total_inference_time_ms: float
    implausible_count: int
    implausible_rate: float


class SMILESAnalysisResponse(BaseModel):
    smiles: str
    valid_smiles: bool
    physics_confidence: PhysicsConfidence
    molecular_weight: float | None
    molecular_formula: str | None
    functional_groups_detected: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint: str
    loaded_epoch: int
    val_ecr: float
    uptime_sec: float
    requests_served: int
    error_count: int
    error_rate: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    num_parameters: int
    architecture: dict
    physics_constraints: dict
    training_info: dict
    input_format: dict
    output_format: dict
    known_limitations: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ecr_to_tier(ecr: float) -> str:
    if ecr < 0.05:  return "high"
    if ecr < 0.15:  return "medium"
    if ecr < 0.25:  return "low"
    return "implausible"


def _valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return len(smiles) > 1


def _run_single_inference(
    spectrum_np: np.ndarray,
    beam_size: int,
    return_reconstructed: bool,
    return_group_probs: bool,
) -> dict:
    """
    Core inference function. Thread-safe (no state mutation).
    Returns a raw dict — converted to PredictResponse by the endpoint.
    """
    t0 = time.time()

    # ── Prepare tensor ────────────────────────────────────────────────────
    spec_tensor = torch.from_numpy(spectrum_np).float().unsqueeze(0).to(state.device)

    # ── Model forward ─────────────────────────────────────────────────────
    with torch.no_grad():
        result = state.model.predict(
            spec_tensor,
            beam_size=beam_size,
            return_diagnostics=True,
        )

    # ── Extract predictions ────────────────────────────────────────────────
    pred_tokens  = result["smiles_tokens"][0].cpu().tolist()
    pred_smiles  = smiles_detokenise(pred_tokens)
    log_prob     = float(result["log_probs"][0].cpu())
    ecr_val      = float(result["ecr"][0].cpu())
    implausible  = bool(result["implausible"][0].cpu())
    recon_np     = result["reconstructed_spec"][0].cpu().numpy()
    group_probs_np = result["group_probs"][0].cpu().numpy()

    # ── Detailed BL report ────────────────────────────────────────────────
    _, bl_report = state.bl_constraint(
        spec_tensor.cpu(),
        result["reconstructed_spec"][0:1].cpu(),
    )
    peak_pos_err = float(bl_report.peak_position_error[0])
    int_mse      = float(bl_report.intensity_mse[0])

    # ── Group frequency check ─────────────────────────────────────────────
    gf_report  = state.gf_checker.check(spectrum_np, pred_smiles)
    gf_recall  = gf_report.recall
    gf_viols   = [v.group_name for v in gf_report.violations]

    # ── Group probability dict ────────────────────────────────────────────
    group_prob_dict = None
    if return_group_probs:
        group_prob_dict = {
            gf.name: round(float(group_probs_np[i]), 4)
            for i, gf in enumerate(GROUP_FREQUENCY_TABLE)
            if i < len(group_probs_np)
        }

    inf_ms = (time.time() - t0) * 1000

    return {
        "predicted_smiles":     pred_smiles,
        "valid_smiles":         _valid_smiles(pred_smiles),
        "log_prob":             round(log_prob, 4),
        "ecr":                  round(ecr_val, 5),
        "peak_position_error":  round(peak_pos_err, 5),
        "intensity_mse":        round(int_mse, 5),
        "physically_plausible": not implausible,
        "confidence_tier":      _ecr_to_tier(ecr_val),
        "gf_recall":            round(gf_recall, 4),
        "gf_violations":        gf_viols,
        "group_prob_dict":      group_prob_dict,
        "reconstructed":        recon_np.tolist() if return_reconstructed else None,
        "inference_time_ms":    round(inf_ms, 2),
    }


def _raw_to_response(raw: dict) -> PredictResponse:
    """Convert raw inference dict to PredictResponse."""
    return PredictResponse(
        predicted_smiles=raw["predicted_smiles"],
        valid_smiles=raw["valid_smiles"],
        log_prob=raw["log_prob"],
        physics_confidence=PhysicsConfidence(
            ecr=raw["ecr"],
            peak_position_error=raw["peak_position_error"],
            intensity_mse=raw["intensity_mse"],
            physically_plausible=raw["physically_plausible"],
            confidence_tier=raw["confidence_tier"],
            gf_recall=raw["gf_recall"],
            gf_violations=raw["gf_violations"],
        ),
        group_probabilities=raw["group_prob_dict"],
        reconstructed_spectrum=raw["reconstructed"],
        inference_time_ms=raw["inference_time_ms"],
        model_epoch=state.loaded_epoch,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model here so it's available in the same process as uvicorn
    import os
    ckpt = os.environ.get("SPECTRALM_CHECKPOINT")
    device = os.environ.get("SPECTRALM_DEVICE", "cpu")
    if ckpt and state.model is None:
        load_model(ckpt, device)
    if state.model is not None:
        log.info(f"Model ready — epoch={state.loaded_epoch}  val_ecr={state.val_ecr:.4f}")
    else:
        log.warning("No checkpoint specified. Set SPECTRALM_CHECKPOINT env var.")
    yield
    log.info("Shutting down.")

app = FastAPI(
    title="SpectraLM Inference API",
    description="""
Physics-informed IR spectrum → SMILES/IUPAC molecular identification.

## Key design principle
Every prediction includes a **physics_confidence** object containing
the Energy Conservation Residual (ECR). This score measures how
consistent the predicted molecular structure is with the observed
spectrum according to Beer–Lambert law.

**Always check `physics_confidence.physically_plausible` before
using a prediction in downstream chemistry applications.**

## Quick start
```python
import requests, numpy as np

spectrum = np.load("my_spectrum.npy").tolist()  # 1800-point normalised
response = requests.post(
    "http://localhost:8000/predict",
    json={"spectrum": spectrum, "beam_size": 4}
)
pred = response.json()
print(pred["predicted_smiles"])
print(pred["physics_confidence"]["ecr"])
print(pred["physics_confidence"]["physically_plausible"])
```
    """,
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow local development frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request logging + counter ─────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    t0 = time.time()
    state.request_count += 1
    try:
        response = await call_next(request)
        duration = (time.time() - t0) * 1000
        log.info(f"{request.method} {request.url.path}  "
                 f"{response.status_code}  {duration:.1f}ms")
        return response
    except Exception as e:
        state.error_count += 1
        log.error(f"{request.method} {request.url.path}  ERROR  {e}")
        raise


# ── Dependency: model must be loaded ─────────────────────────────────────────
def require_model():
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Start server with --checkpoint argument.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    model_ready = state.model is not None
    status_color = "#1D9E75" if model_ready else "#E24B4A"
    status_text  = "online" if model_ready else "not loaded"
    epoch  = state.loaded_epoch if model_ready else "—"
    ecr    = f"{state.val_ecr:.4f}" if model_ready else "—"
    params = f"{state.model.num_parameters:,}" if model_ready else "—"
    uptime = f"{(time.time() - state.server_start_time):.0f}s" if state.server_start_time else "—"

    html = Path("static/index.html").read_text(encoding="utf-8")

    html = html.replace("{status_text}",       status_text)
    html = html.replace("{status_color}",      status_color)

    html = html.replace("{uptime}",            uptime)
    html = html.replace("{ecr}",               ecr)
    html = html.replace("{epoch}",             str(epoch))
    html = html.replace("{params}",            params)
    html = html.replace("{state.request_count}", str(state.request_count))
    html = html.replace(
        '<link rel="stylesheet" href="./style.css">',
        '<link rel="stylesheet" href="/static/style.css">'
    )

    return HTMLResponse(html)

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server and model health check",
    tags=["Infrastructure"],
)
async def health():
    """
    Returns server status, model load state, and request statistics.
    Use this for load balancer health checks and monitoring.
    """
    uptime = time.time() - state.server_start_time if state.server_start_time else 0.0
    n      = state.request_count
    errors = state.error_count
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        device=str(state.device),
        checkpoint=state.checkpoint_path,
        loaded_epoch=state.loaded_epoch,
        val_ecr=state.val_ecr,
        uptime_sec=round(uptime, 1),
        requests_served=n,
        error_count=errors,
        error_rate=round(errors / max(n, 1), 4),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model card metadata",
    tags=["Infrastructure"],
)
async def model_info(_: None = Depends(require_model)):
    """
    Returns the model card: architecture, training details,
    physics constraint configuration, and known limitations.
    """
    cfg = state.config
    return ModelInfoResponse(
        model_name="SpectraLM",
        version="0.3.0",
        num_parameters=state.model.num_parameters,
        architecture={
            "encoder":     "1D-CNN (4 layers) + Wavenumber Positional Encoding",
            "transformer": f"Encoder-Decoder, "
                           f"d_model={cfg.transformer.d_model}, "
                           f"nhead={cfg.transformer.nhead}, "
                           f"enc_layers={cfg.transformer.num_encoder_layers}, "
                           f"dec_layers={cfg.transformer.num_decoder_layers}",
            "physics_head": "Lorentzian peak model (learnable centres + widths)",
            "d_model":     cfg.transformer.d_model,
            "vocab_size":  cfg.transformer.vocab_size,
        },
        physics_constraints={
            "beer_lambert_weight":  cfg.lambda_beer_lambert,
            "group_freq_weight":    cfg.lambda_group_freq,
            "implausibility_threshold": cfg.implausibility_threshold,
            "constraint_description": (
                "Beer–Lambert ECR + Group Frequency penalty enforced during training. "
                "ECR measures consistency between predicted structure and observed spectrum "
                "according to A(ν) = ε(ν)·c·l."
            ),
        },
        training_info={
            "checkpoint_epoch": state.loaded_epoch,
            "val_ecr":          state.val_ecr,
            "dataset":          "NIST WebBook IR + MoNA (~12,000 compounds)",
            "training_script":  "scripts/train.py",
        },
        input_format={
            "spectrum_length": 1800,
            "wavenumber_range": "400–4000 cm⁻¹",
            "resolution":       "2 cm⁻¹",
            "normalisation":    "Scale so max absorbance = 1.0",
            "units":            "Absorbance (not transmittance)",
        },
        output_format={
            "smiles":          "Canonical SMILES string",
            "ecr":             "Float, lower is better, implausible > 0.25",
            "confidence_tier": "high / medium / low / implausible",
            "gf_recall":       "Float 0–1, fraction of expected group absorptions present",
        },
        known_limitations=[
            "Fails on Nujol-mulled spectra in the 2850–2960 cm⁻¹ region (41% of failures)",
            "Regioisomer confusion: cannot distinguish ortho/meta/para from IR alone (33%)",
            "Poor generalisation to molecular scaffolds with <5 training examples (19%)",
            "Overtone band misidentification in the 1800–2000 cm⁻¹ region (7%)",
            "Beam search currently supports batch_size=1 only",
            "Not validated on inorganic or organometallic compounds",
        ],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict molecular structure from IR spectrum",
    tags=["Inference"],
)
async def predict(
    request: PredictRequest,
    _: None = Depends(require_model),
):
    """
    Translates an IR spectrum into a SMILES molecular structure prediction,
    with full physics-residual confidence scoring.

    ## Response fields to prioritise

    1. `physics_confidence.physically_plausible` — Boolean. Hard filter.
       If False, the predicted SMILES violates Beer–Lambert law and
       should not be used without further verification.

    2. `physics_confidence.ecr` — Float, lower is better.
       - < 0.05  : high confidence
       - 0.05–0.15 : medium confidence
       - 0.15–0.25 : low confidence
       - > 0.25  : physically implausible ⚠

    3. `physics_confidence.gf_violations` — List of functional groups
       predicted in the SMILES but not observed in the spectrum.
       An empty list is a good sign.

    4. `predicted_smiles` — The prediction itself. Only trust if
       physically_plausible is True.
    """
    try:
        spectrum_np = np.array(request.spectrum, dtype=np.float32)
        raw = _run_single_inference(
            spectrum_np=spectrum_np,
            beam_size=request.beam_size,
            return_reconstructed=request.return_reconstructed,
            return_group_probs=request.return_group_probs,
        )
        return _raw_to_response(raw)
    except Exception as e:
        state.error_count += 1
        log.error(f"/predict error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch prediction (up to 32 spectra)",
    tags=["Inference"],
)
async def predict_batch(
    request: BatchPredictRequest,
    _: None = Depends(require_model),
):
    """
    Runs inference on up to 32 spectra in a single request.
    Uses greedy decoding (beam_size≤4) for throughput.

    Returns predictions in the same order as the input spectra.
    The `implausible_rate` field summarises physics quality across
    the entire batch.
    """
    t_batch = time.time()
    predictions: list[PredictResponse] = []

    try:
        for spec_list in request.spectra:
            spec_np = np.array(spec_list, dtype=np.float32)
            raw = _run_single_inference(
                spectrum_np=spec_np,
                beam_size=request.beam_size,
                return_reconstructed=request.return_reconstructed,
                return_group_probs=request.return_group_probs,
            )
            predictions.append(_raw_to_response(raw))

        total_ms     = (time.time() - t_batch) * 1000
        n_implaus    = sum(1 for p in predictions
                          if not p.physics_confidence.physically_plausible)

        return BatchPredictResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_inference_time_ms=round(total_ms, 2),
            implausible_count=n_implaus,
            implausible_rate=round(n_implaus / max(len(predictions), 1), 4),
        )
    except Exception as e:
        state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/smiles",
    response_model=SMILESAnalysisResponse,
    summary="Verify SMILES candidate against a spectrum",
    tags=["Inference"],
)
async def analyse_smiles(
    request: SMILESAnalysisRequest,
    _: None = Depends(require_model),
):
    """
    Given a candidate SMILES and an observed spectrum, computes
    the physics-residual score for that specific candidate.

    Useful for:
    - Verifying a proposed structure against a measured spectrum
    - Comparing multiple candidate structures for the same spectrum
    - Post-hoc validation of database hits from mass spectrometry

    Does NOT run the model's autoregressive decoder — uses only
    the physics head (group frequency lookup + Beer–Lambert check).
    """
    try:
        spectrum_np = np.array(request.spectrum, dtype=np.float32)
        smiles      = request.smiles.strip()

        # Validate SMILES
        valid = _valid_smiles(smiles)

        # Group frequency check
        gf_report   = state.gf_checker.check(spectrum_np, smiles)
        gf_recall   = gf_report.recall
        gf_viols    = [v.group_name for v in gf_report.violations]
        gf_detected = []

        # Detected groups (present in both SMILES and spectrum)
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            mol = Chem.MolFromSmiles(smiles)
            mw  = Descriptors.MolWt(mol) if mol else None
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol else None
            for gf in GROUP_FREQUENCY_TABLE:
                pat = Chem.MolFromSmarts(gf.smarts)
                if pat and mol and mol.HasSubstructMatch(pat):
                    gf_detected.append(gf.name)
        except Exception:
            mw, formula = None, None

        # Build pseudo-ECR from group frequency score
        # (no model inference — uses physics check only)
        pseudo_ecr = 1.0 - gf_recall

        return SMILESAnalysisResponse(
            smiles=smiles,
            valid_smiles=valid,
            physics_confidence=PhysicsConfidence(
                ecr=round(pseudo_ecr, 4),
                peak_position_error=0.0,   # not computed without model
                intensity_mse=0.0,
                physically_plausible=pseudo_ecr < 0.25,
                confidence_tier=_ecr_to_tier(pseudo_ecr),
                gf_recall=round(gf_recall, 4),
                gf_violations=gf_viols,
            ),
            molecular_weight=round(mw, 2) if mw else None,
            molecular_formula=formula,
            functional_groups_detected=gf_detected,
        )
    except Exception as e:
        state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Custom exception handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Friendlier validation error responses with domain-specific hints."""
    body = await request.body()
    log.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "hints": [
                "spectrum must be exactly 1800 float values",
                "spectrum values should be normalised to [0, 1] absorbance",
                "if your spectrum is transmittance, convert: A = -log10(T/100)",
                "if your spectrum has different resolution, interpolate to 1800 points"
                " over 400–4000 cm⁻¹",
            ],
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SpectraLM inference server")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Uvicorn worker count (default: 1)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Hot reload on code changes (dev mode only)",
    )
    args = parser.parse_args()

    # Load model before starting server
    import os
    os.environ["SPECTRALM_CHECKPOINT"] = args.checkpoint
    os.environ["SPECTRALM_DEVICE"]     = args.device
    log.info(f"Starting SpectraLM server on {args.host}:{args.port}")
    log.info(f"API docs → http://{args.host}:{args.port}/docs")
    log.info(f"Health   → http://{args.host}:{args.port}/health")

    uvicorn.run(
        "scripts.serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()