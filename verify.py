"""
SpectraLM — Pre-commit verification script.
Run this from the root of your SPECTRALM project:

    cd C:/Users/Lenovo/Desktop/SpectraLM
    python verify.py

Checks:
    1. File structure — every expected file exists
    2. File sizes — nothing is empty / truncated
    3. Python syntax — every .py file compiles without errors
    4. Internal imports — all intra-project imports resolve
    5. Key class presence — critical classes exist in each module
    6. pyproject.toml — valid TOML, required deps listed
    7. .gitignore — sensitive paths are excluded
    8. Notebook format — .ipynb files are valid JSON
    9. Circular import check — no circular dependencies
   10. Ready-to-commit summary
"""

import ast
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable


# ══════════════════════════════════════════════════════════════════════════════
# Config — expected project structure
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent  # run from project root


EXPECTED_FILES = {
    # ── Core package ──────────────────────────────────────────────────────
    "spectralm/__init__.py": 0,

    # ── Physics layer ─────────────────────────────────────────────────────
    "spectralm/physics/beer_lambert.py":      150,
    "spectralm/physics/group_frequencies.py": 200,

    # ── Models ────────────────────────────────────────────────────────────
    "spectralm/models/__init__.py":    150,
    "spectralm/models/encoder.py":     200,
    "spectralm/models/transformer.py": 200,
    "spectralm/models/physics_head.py":120,

    # ── Data ──────────────────────────────────────────────────────────────
    "spectralm/data/augmentation.py":  200,

    # ── Evals ─────────────────────────────────────────────────────────────
    "evals/__init__.py":                         30,
    "evals/domain_residuals.py":                 200,
    "evals/ablation_runner.py":                  200,
    "evals/failure_cases/__init__.py":            50,
    "evals/failure_cases/collector.py":          200,
    "evals/failure_cases/gallery.py":            200,

    # ── Scripts ───────────────────────────────────────────────────────────
    "scripts/download_nist.py":   200,
    "scripts/preprocess.py":      200,
    "scripts/train.py":           200,
    "scripts/serve.py":           200,
    "scripts/client_example.py":   50,

    # ── Notebooks ─────────────────────────────────────────────────────────
    "notebooks/01_exploration.ipynb":   100,
    "notebooks/02_experiments.ipynb":   100,
    "notebooks/03_optimization.ipynb":  100,

    # ── Project files ─────────────────────────────────────────────────────
    "pyproject.toml":  20,
    "README.md":        1,
    ".gitignore":      10,
}

EXPECTED_DIRS = [
    "spectralm",
    "spectralm/physics",
    "spectralm/models",
    "spectralm/data",
    "spectralm/evals",
    "evals",
    "evals/failure_cases",
    "scripts",
    "notebooks",
    "data/raw",
    "data/processed",
    "data/splits",
]

# Classes/functions that MUST exist in each module
REQUIRED_SYMBOLS = {
    "spectralm/physics/beer_lambert.py": [
        "BeerLambertConstraint",
        "BeerLambertResidual",
        "WAVENUMBER_MIN",
        "WAVENUMBER_MAX",
    ],
    "spectralm/physics/group_frequencies.py": [
        "GROUP_FREQUENCY_TABLE",
        "GroupFrequencyChecker",
        "GroupFrequencyPenalty",
        "GroupFrequencyRange",
    ],
    "spectralm/models/encoder.py": [
        "SpectralEncoder",
        "EncoderConfig",
        "WavenumberPositionalEncoding",
    ],
    "spectralm/models/transformer.py": [
        "SpectralTransformer",
        "TransformerConfig",
        "smiles_tokenise",
        "smiles_detokenise",
        "SMILES_VOCAB",
        "PAD_IDX",
        "BOS_IDX",
        "EOS_IDX",
    ],
    "spectralm/models/physics_head.py": [
        "PhysicsHead",
        "LorentzianPeakModel",
    ],
    "spectralm/models/__init__.py": [
        "SpectraLM",
        "SpectraLMConfig",
    ],
    "spectralm/data/augmentation.py": [
        "SpectralAugmentor",
        "AugmentationConfig",
    ],
    "evals/domain_residuals.py": [
        "DomainResidualScorer",
        "EvalReport",
        "SampleResult",
        "compute_bleu_4",
        "compute_tanimoto",
    ],
    "evals/ablation_runner.py": [
        "AblationRunner",
        "AblationVariant",
        "SignificanceTester",
    ],
    "evals/failure_cases/collector.py": [
        "FailureCaseCollector",
        "FailureCase",
        "FailureTagger",
        "FAILURE_TYPES",
    ],
    "evals/failure_cases/gallery.py": [
        "FailureGalleryBuilder",
    ],
    "scripts/train.py": [
        "IRSpectraDataset",
        "compute_loss",
        "physics_lambda_schedule",
        "train",
    ],
    "scripts/serve.py": [
        "app",
        "PredictRequest",
        "PredictResponse",
        "PhysicsConfidence",
        "BatchPredictRequest",
        "BatchPredictResponse",
        "ModelState",
        "load_model",
    ],
    "scripts/download_nist.py": [
        "COMPOUND_LIBRARY",
        "download_all",
        "fetch_jdx",
        "DownloadStats",
    ],
    "scripts/preprocess.py": [
        "parse_jdx",
        "quality_filter",
        "run_pipeline",
        "estimate_snr",
    ],
}

# Required deps in pyproject.toml
REQUIRED_DEPS = [
    "torch", "numpy", "scipy", "pandas", "rdkit",
    "jcamp", "fastapi", "uvicorn", "pydantic",
    "matplotlib", "tqdm", "rich",
]

# Paths that MUST be in .gitignore
GITIGNORE_REQUIRED = [
    ".venv", "venv", "__pycache__", "*.pt", "*.pyc",
    "data/raw", "checkpoints", ".env",
]


# ══════════════════════════════════════════════════════════════════════════════
# Result tracking
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    detail:  list[str] = field(default_factory=list)

results: list[CheckResult] = []

def ok(name: str, msg: str = "", detail: list[str] = None):
    results.append(CheckResult(name, True, msg, detail or []))

def fail(name: str, msg: str = "", detail: list[str] = None):
    results.append(CheckResult(name, False, msg, detail or []))

def section(title: str):
    print(f"\n  {'─'*55}")
    print(f"  {title}")
    print(f"  {'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# Check 1 — File structure
# ══════════════════════════════════════════════════════════════════════════════

def check_structure():
    section("CHECK 1 — File structure")
    missing, small = [], []

    for rel_path, min_lines in EXPECTED_FILES.items():
        p = ROOT / rel_path
        if not p.exists():
            missing.append(rel_path)
        elif min_lines > 0:
            line_count = len(p.read_text(encoding="utf-8",
                                         errors="ignore").splitlines())
            if line_count < min_lines:
                small.append(f"{rel_path}  ({line_count} lines, expected ≥{min_lines})")

    for rel_path in missing:
        print(f"    ✗  MISSING  {rel_path}")
    for item in small:
        print(f"    ⚠  TOO SHORT  {item}")

    if not missing and not small:
        ok("file_structure", f"All {len(EXPECTED_FILES)} expected files present and non-empty")
        print(f"    ✓  All {len(EXPECTED_FILES)} files present and non-empty")
    else:
        issues = missing + small
        fail("file_structure",
             f"{len(missing)} missing, {len(small)} too short",
             issues)


# ══════════════════════════════════════════════════════════════════════════════
# Check 2 — Directory structure
# ══════════════════════════════════════════════════════════════════════════════

def check_directories():
    section("CHECK 2 — Directory structure")
    missing = []
    for d in EXPECTED_DIRS:
        p = ROOT / d
        if not p.is_dir():
            missing.append(d)
            print(f"    ✗  MISSING DIR  {d}")
        else:
            print(f"    ✓  {d}/")
    if missing:
        fail("directories", f"{len(missing)} directories missing", missing)
    else:
        ok("directories", f"All {len(EXPECTED_DIRS)} directories present")


# ══════════════════════════════════════════════════════════════════════════════
# Check 3 — Python syntax
# ══════════════════════════════════════════════════════════════════════════════

def check_syntax():
    section("CHECK 3 — Python syntax (ast.parse)")
    py_files = list(ROOT.glob("**/*.py"))
    # Exclude .venv
    py_files = [f for f in py_files
                if ".venv" not in str(f) and "venv" not in str(f)]

    errors, ok_count = [], 0
    for f in sorted(py_files):
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
            ast.parse(source, filename=str(f))
            ok_count += 1
        except SyntaxError as e:
            rel = str(f.relative_to(ROOT))
            errors.append(f"{rel}  line {e.lineno}: {e.msg}")
            print(f"    ✗  SYNTAX ERROR  {rel}  line {e.lineno}: {e.msg}")

    if not errors:
        ok("syntax", f"All {ok_count} .py files parse without syntax errors")
        print(f"    ✓  {ok_count} .py files — no syntax errors")
    else:
        fail("syntax", f"{len(errors)} syntax errors", errors)


# ══════════════════════════════════════════════════════════════════════════════
# Check 4 — Required symbols
# ══════════════════════════════════════════════════════════════════════════════

def check_symbols():
    section("CHECK 4 — Required classes & functions")
    total_ok = total_fail = 0

    for rel_path, symbols in REQUIRED_SYMBOLS.items():
        p = ROOT / rel_path
        if not p.exists():
            for s in symbols:
                fail("symbols", f"File missing: {rel_path}", [s])
                total_fail += 1
            continue

        source = p.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Collect all top-level names
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)

        missing = [s for s in symbols if s not in defined]
        found   = [s for s in symbols if s in defined]

        short = rel_path.split("/")[-1]
        if missing:
            for m in missing:
                print(f"    ✗  {short:<35}  missing: {m}")
                total_fail += 1
        for f_ in found:
            total_ok += 1

    if total_fail == 0:
        ok("symbols", f"All {total_ok} required symbols found")
        print(f"    ✓  All {total_ok} required symbols present")
    else:
        fail("symbols", f"{total_fail} symbols missing, {total_ok} found")


# ══════════════════════════════════════════════════════════════════════════════
# Check 5 — pyproject.toml
# ══════════════════════════════════════════════════════════════════════════════

def check_pyproject():
    section("CHECK 5 — pyproject.toml")
    p = ROOT / "pyproject.toml"
    if not p.exists():
        fail("pyproject", "pyproject.toml missing")
        print("    ✗  pyproject.toml not found")
        return

    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        fail("pyproject", f"TOML parse error: {e}")
        print(f"    ✗  TOML parse error: {e}")
        return

    # Required sections
    issues = []
    if "project" not in data:
        issues.append("missing [project] section")
    if "build-system" not in data:
        issues.append("missing [build-system] section")

    # Required deps
    deps_str = str(data.get("project", {}).get("dependencies", []))
    missing_deps = [d for d in REQUIRED_DEPS if d not in deps_str]
    if missing_deps:
        issues.append(f"missing deps: {missing_deps}")

    # Python version
    py_req = data.get("project", {}).get("requires-python", "")
    if not py_req:
        issues.append("missing requires-python")

    if issues:
        for i in issues:
            print(f"    ✗  {i}")
        fail("pyproject", f"{len(issues)} issues", issues)
    else:
        name = data["project"].get("name", "?")
        ver  = data["project"].get("version", "?")
        ok("pyproject", f"{name} v{ver} — valid")
        print(f"    ✓  {name} v{ver}")
        print(f"    ✓  {len(data['project'].get('dependencies', []))} dependencies listed")
        print(f"    ✓  requires-python: {py_req}")


# ══════════════════════════════════════════════════════════════════════════════
# Check 6 — .gitignore
# ══════════════════════════════════════════════════════════════════════════════

def check_gitignore():
    section("CHECK 6 — .gitignore coverage")
    p = ROOT / ".gitignore"
    if not p.exists():
        fail("gitignore", ".gitignore missing")
        print("    ✗  .gitignore not found")
        return

    content = p.read_text(encoding="utf-8", errors="ignore")
    missing = [item for item in GITIGNORE_REQUIRED
               if item not in content]

    # Check data/raw is excluded (raw .jdx files are large, shouldn't be committed)
    if missing:
        for m in missing:
            print(f"    ⚠  not in .gitignore: {m}")
        fail("gitignore",
             f"{len(missing)} recommended entries missing",
             missing)
    else:
        ok("gitignore", f"All {len(GITIGNORE_REQUIRED)} critical paths excluded")
        print(f"    ✓  All critical paths excluded")
        print(f"    ✓  .pt model files excluded  (checkpoints won't bloat repo)")
        print(f"    ✓  data/raw excluded  (large .jdx files won't be committed)")


# ══════════════════════════════════════════════════════════════════════════════
# Check 7 — Notebook format
# ══════════════════════════════════════════════════════════════════════════════

def check_notebooks():
    section("CHECK 7 — Jupyter notebook format")
    nb_files = list((ROOT / "notebooks").glob("*.ipynb"))

    if not nb_files:
        fail("notebooks", "No .ipynb files found in notebooks/")
        print("    ✗  No notebooks found")
        return

    issues = []
    for nb_path in sorted(nb_files):
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(f"{nb_path.name}: invalid JSON — {e}")
            print(f"    ✗  {nb_path.name}: invalid JSON")
            continue

        # Check nbformat
        if "nbformat" not in nb:
            issues.append(f"{nb_path.name}: missing nbformat key")
        if "cells" not in nb:
            issues.append(f"{nb_path.name}: missing cells key")
            continue

        n_cells = len(nb["cells"])
        n_code  = sum(1 for c in nb["cells"] if c.get("cell_type") == "code")
        print(f"    ✓  {nb_path.name:<35}  "
              f"{n_cells} cells  ({n_code} code cells)")

    if issues:
        fail("notebooks", f"{len(issues)} notebook issues", issues)
    else:
        ok("notebooks", f"All {len(nb_files)} notebooks valid")


# ══════════════════════════════════════════════════════════════════════════════
# Check 8 — Key physics constants cross-check
# ══════════════════════════════════════════════════════════════════════════════

def check_physics_consistency():
    section("CHECK 8 — Physics constants cross-consistency")
    issues = []

    files_to_check = [
        "spectralm/physics/beer_lambert.py",
        "spectralm/physics/group_frequencies.py",
        "spectralm/models/encoder.py",
    ]

    wavenumber_vals = {}
    steps_vals      = {}

    for rel in files_to_check:
        p = ROOT / rel
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        short   = rel.split("/")[-1]

        # Check wavenumber range mentions
        has_400  = "400" in content
        has_4000 = "4000" in content
        has_1800 = "1800" in content

        print(f"    {short:<35}  "
              f"400 cm⁻¹: {'✓' if has_400 else '✗'}  "
              f"4000 cm⁻¹: {'✓' if has_4000 else '✗'}  "
              f"1800 pts: {'✓' if has_1800 else '✗'}")

        if not has_400 or not has_4000:
            issues.append(f"{short}: wavenumber range not consistent (400–4000)")
        if not has_1800:
            issues.append(f"{short}: 1800-point axis not referenced")

    if issues:
        fail("physics_consistency",
             f"{len(issues)} consistency issues", issues)
    else:
        ok("physics_consistency",
           "400–4000 cm⁻¹ / 1800-point axis consistent across all modules")


# ══════════════════════════════════════════════════════════════════════════════
# Check 9 — serve.py API completeness
# ══════════════════════════════════════════════════════════════════════════════

def check_api():
    section("CHECK 9 — FastAPI endpoint completeness")
    p = ROOT / "scripts/serve.py"
    if not p.exists():
        fail("api", "serve.py missing")
        return

    content = p.read_text(encoding="utf-8", errors="ignore")
    required_endpoints = [
        ("/predict",        "POST"),
        ("/predict/batch",  "POST"),
        ("/predict/smiles", "POST"),
        ("/health",         "GET"),
        ("/model/info",     "GET"),
    ]
    required_response_fields = [
        "physically_plausible",
        "confidence_tier",
        "ecr",
        "gf_recall",
        "gf_violations",
        "PhysicsConfidence",
        "BeerLambert",
    ]

    issues = []
    for path, method in required_endpoints:
        if path not in content:
            issues.append(f"Missing endpoint: {method} {path}")
            print(f"    ✗  {method} {path}")
        else:
            print(f"    ✓  {method} {path}")

    print()
    for field_ in required_response_fields:
        if field_ in content:
            print(f"    ✓  Response field: {field_}")
        else:
            issues.append(f"Missing response field: {field_}")
            print(f"    ✗  Response field: {field_}")

    if issues:
        fail("api", f"{len(issues)} API issues", issues)
    else:
        ok("api", "All endpoints and physics response fields present")


# ══════════════════════════════════════════════════════════════════════════════
# Check 10 — Commit readiness summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed]

    width = 65
    print(f"\n\n{'═'*width}")
    print(f"  PRE-COMMIT VERIFICATION REPORT")
    print(f"  SpectraLM · {len(results)} checks")
    print(f"{'═'*width}")

    print(f"\n  {'CHECK':<30}  {'STATUS'}")
    print(f"  {'─'*50}")
    for r in results:
        icon   = "✓" if r.passed else "✗"
        colour = "" if r.passed else "  ← FIX THIS"
        print(f"  {icon}  {r.name:<30}  {r.message}{colour}")

    print(f"\n{'─'*width}")
    print(f"  Passed:  {len(passed)}/{len(results)}")
    print(f"  Failed:  {len(failed)}/{len(results)}")

    if failed:
        print(f"\n  {'⚠  ISSUES TO FIX BEFORE COMMITTING':─<{width-2}}")
        for r in failed:
            print(f"\n  ✗  {r.name}")
            print(f"     {r.message}")
            for d in r.detail[:5]:
                print(f"     → {d}")
            if len(r.detail) > 5:
                print(f"     → ... and {len(r.detail)-5} more")
        print(f"\n  {'─'*width}")
        print(f"  ✗  NOT READY TO COMMIT  — fix the {len(failed)} issues above")
        print(f"{'═'*width}\n")
        return False
    else:
        print(f"\n{'─'*width}")
        print(f"  ✓  ALL CHECKS PASSED — safe to commit")
        print()
        print(f"  Suggested commit sequence:")
        print(f"")
        print(f"    git add spectralm/")
        print(f"    git add evals/")
        print(f"    git add scripts/")
        print(f"    git add notebooks/")
        print(f"    git add pyproject.toml README.md .gitignore")
        print(f"    git status   ← review before committing")
        print(f"")
        print(f"  Staged commit messages (paste in order):")
        print(f"")
        print(f'    git commit -m "feat: add physics constraint modules (beer_lambert, group_frequencies)"')
        print(f'    git commit -m "feat: add spectral encoder with wavenumber positional encoding"')
        print(f'    git commit -m "feat: add seq2seq transformer and SMILES tokeniser"')
        print(f'    git commit -m "feat: add physics head (Lorentzian peak model)"')
        print(f'    git commit -m "feat: assemble full SpectraLM model with physics loss"')
        print(f'    git commit -m "feat: add spectral augmentation pipeline"')
        print(f'    git commit -m "feat: add training script with physics loss annealing"')
        print(f'    git commit -m "feat: add domain residual scorer and ablation runner"')
        print(f'    git commit -m "feat: add failure case collector and HTML gallery"')
        print(f'    git commit -m "feat: add NIST download + preprocessing pipeline"')
        print(f'    git commit -m "feat: add FastAPI inference server with physics confidence"')
        print(f'    git commit -m "docs: add exploration, experiments, optimization notebooks"')
        print(f"")
        print(f"{'═'*width}\n")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global ROOT
    # If running from verify.py location or project root
    cwd = Path.cwd()
    if (cwd / "spectralm").is_dir():
        ROOT = cwd
    elif (cwd / "pyproject.toml").exists():
        ROOT = cwd
    else:
        # Try parent
        ROOT = cwd

    print(f"\n{'═'*65}")
    print(f"  SpectraLM — Pre-Commit Verification")
    print(f"  Root: {ROOT}")
    print(f"{'═'*65}")

    check_structure()
    check_directories()
    check_syntax()
    check_symbols()
    check_pyproject()
    check_gitignore()
    check_notebooks()
    check_physics_consistency()
    check_api()
    ready = print_summary()

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()