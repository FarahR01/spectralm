"""
scripts/client_example.py

Example client for the SpectraLM inference API.
Demonstrates all endpoints with realistic usage patterns.

Usage:
    # First start the server:
    python scripts/serve.py --checkpoint checkpoints/best_model.pt

    # Then run examples:
    python scripts/client_example.py
"""

from __future__ import annotations

import json
import numpy as np
import requests

BASE_URL = "http://localhost:8000"


def make_test_spectrum() -> list[float]:
    """
    Generate a synthetic ethanol-like spectrum for testing.
    In production: load your real .jdx file via scripts/preprocess.py
    and use the normalised numpy array directly.
    """
    wn  = np.linspace(400, 4000, 1800)
    spec = np.zeros(1800)
    # O-H stretch (alcohol) — broad, ~3340 cm⁻¹
    spec += 0.85 * np.exp(-((wn - 3340) / 120) ** 2)
    # C-H stretch — ~2940 cm⁻¹
    spec += 0.60 * np.exp(-((wn - 2940) / 50) ** 2)
    spec += 0.45 * np.exp(-((wn - 2880) / 40) ** 2)
    # C-O stretch — ~1050 cm⁻¹
    spec += 0.75 * np.exp(-((wn - 1050) / 40) ** 2)
    # O-H bend — ~1380 cm⁻¹
    spec += 0.30 * np.exp(-((wn - 1380) / 30) ** 2)
    # Add realistic noise
    spec += np.random.normal(0, 0.008, spec.shape)
    spec = np.clip(spec, 0, None)
    spec /= spec.max()
    return spec.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Example 1: Single prediction
# ══════════════════════════════════════════════════════════════════════════════

def example_single_predict():
    print("\n" + "═" * 60)
    print("Example 1: Single prediction")
    print("═" * 60)

    spectrum = make_test_spectrum()

    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "spectrum":           spectrum,
            "beam_size":          4,
            "return_group_probs": True,
            "return_reconstructed": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    pred = response.json()

    print(f"\nPredicted SMILES:  {pred['predicted_smiles']}")
    print(f"Valid SMILES:      {pred['valid_smiles']}")
    print(f"Log-probability:   {pred['log_prob']:.3f}")
    print(f"Inference time:    {pred['inference_time_ms']:.1f} ms")

    pc = pred["physics_confidence"]
    print(f"\nPhysics Confidence:")
    print(f"  ECR:                {pc['ecr']:.5f}")
    print(f"  Confidence tier:    {pc['confidence_tier'].upper()}")
    print(f"  Physically plausible: {pc['physically_plausible']}")
    print(f"  GF Recall:          {pc['gf_recall']:.3f}")

    if pc["gf_violations"]:
        print(f"  ⚠ GF Violations:")
        for v in pc["gf_violations"]:
            print(f"    - {v}")
    else:
        print(f"  ✓ No group frequency violations")

    if pred.get("group_probabilities"):
        print(f"\nTop functional group confidences:")
        sorted_groups = sorted(
            pred["group_probabilities"].items(),
            key=lambda x: x[1], reverse=True
        )
        for name, prob in sorted_groups[:5]:
            bar = "█" * int(prob * 20)
            print(f"  {name[:40]:<40}  {bar:<20}  {prob:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Example 2: Batch prediction
# ══════════════════════════════════════════════════════════════════════════════

def example_batch_predict():
    print("\n" + "═" * 60)
    print("Example 2: Batch prediction (5 spectra)")
    print("═" * 60)

    spectra = [make_test_spectrum() for _ in range(5)]

    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"spectra": spectra, "beam_size": 1},
        timeout=60,
    )
    response.raise_for_status()
    batch = response.json()

    print(f"\nBatch size:         {batch['batch_size']}")
    print(f"Total time:         {batch['total_inference_time_ms']:.1f} ms")
    print(f"Avg per spectrum:   {batch['total_inference_time_ms']/batch['batch_size']:.1f} ms")
    print(f"Implausible count:  {batch['implausible_count']}")
    print(f"Implausible rate:   {batch['implausible_rate']:.1%}")
    print()

    for i, pred in enumerate(batch["predictions"]):
        pc     = pred["physics_confidence"]
        flag   = "⚠" if not pc["physically_plausible"] else "✓"
        tier   = pc["confidence_tier"].upper()
        print(f"  [{flag}] #{i+1}  "
              f"{pred['predicted_smiles'][:30]:<30}  "
              f"ECR={pc['ecr']:.4f}  {tier}")


# ══════════════════════════════════════════════════════════════════════════════
# Example 3: SMILES verification
# ══════════════════════════════════════════════════════════════════════════════

def example_smiles_verification():
    print("\n" + "═" * 60)
    print("Example 3: SMILES candidate verification")
    print("═" * 60)

    spectrum  = make_test_spectrum()
    candidates = [
        ("CCO",          "Ethanol (correct)"),
        ("CC(=O)C",      "Acetone (wrong — no O-H)"),
        ("CCCC",         "Butane (very wrong)"),
        ("CC(O)CC",      "2-Butanol (plausible isomer)"),
    ]

    print(f"\n{'Candidate':<25}  {'Label':<28}  "
          f"{'ECR':>6}  {'GF Recall':>10}  {'Plausible':>10}")
    print("─" * 85)

    for smiles, label in candidates:
        resp = requests.post(
            f"{BASE_URL}/predict/smiles",
            json={"smiles": smiles, "spectrum": spectrum},
            timeout=15,
        )
        resp.raise_for_status()
        r  = resp.json()
        pc = r["physics_confidence"]
        flag = "✓" if pc["physically_plausible"] else "✗"
        print(f"  {smiles:<23}  {label:<28}  "
              f"{pc['ecr']:>6.4f}  {pc['gf_recall']:>10.3f}  "
              f"{flag} {pc['confidence_tier']}")


# ══════════════════════════════════════════════════════════════════════════════
# Example 4: Health check + model info
# ══════════════════════════════════════════════════════════════════════════════

def example_health_and_info():
    print("\n" + "═" * 60)
    print("Example 4: Health check + model info")
    print("═" * 60)

    health = requests.get(f"{BASE_URL}/health", timeout=5).json()
    print(f"\nStatus:          {health['status'].upper()}")
    print(f"Model loaded:    {health['model_loaded']}")
    print(f"Device:          {health['device']}")
    print(f"Checkpoint epoch:{health['loaded_epoch']}")
    print(f"Val ECR:         {health['val_ecr']:.4f}")
    print(f"Uptime:          {health['uptime_sec']:.0f}s")
    print(f"Requests served: {health['requests_served']}")

    info = requests.get(f"{BASE_URL}/model/info", timeout=5).json()
    print(f"\nModel:           {info['model_name']} v{info['version']}")
    print(f"Parameters:      {info['num_parameters']:,}")
    print(f"\nPhysics constraints:")
    for k, v in info["physics_constraints"].items():
        if k != "constraint_description":
            print(f"  {k}: {v}")
    print(f"\nKnown limitations:")
    for lim in info["known_limitations"]:
        print(f"  • {lim}")


# ══════════════════════════════════════════════════════════════════════════════
# Run all examples
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SpectraLM API — Client Examples")
    print(f"Server: {BASE_URL}")

    try:
        requests.get(f"{BASE_URL}/health", timeout=3).raise_for_status()
    except Exception:
        print(f"\n❌ Server not running at {BASE_URL}")
        print("   Start it first: python scripts/serve.py "
              "--checkpoint checkpoints/best_model.pt")
        exit(1)

    example_health_and_info()
    example_single_predict()
    example_batch_predict()
    example_smiles_verification()

    print("\n\n✓ All examples complete.")
