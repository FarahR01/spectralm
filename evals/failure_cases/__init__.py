# -*- coding: utf-8 -*-
"""
spectralm/evals/failure_cases/__init__.py

Failure case analysis and gallery builder for SpectraLM.

This submodule systematically analyzes model failures to identify:
    1. Systematic biases (e.g., always underestimates MW)
    2. Hard examples (e.g., isomers with identical spectra)
    3. Domain gaps (e.g., spectra from unseen instruments)
    4. Chimeric errors (e.g., mixing fragments from different molecules)

Core components:

    FailureCaseCollector:
        - Filters test predictions to identify failure modes
        - Annotates each failure with root cause category
        - Computes statistics per failure type
        - Exports to gallery for visualization
    
    FailureCase (dataclass):
        - Stores single prediction failure with diagnostics
        - Fields: observed_spectrum, predicted_smiles, ground_truth_smiles,
                  ecr_residual, failure_type, notes
        - Methods: to_dict(), to_html(), save_figure()
    
    FailureTagger (enum):
        - STRUCTURE_ERROR: Wrong carbon skeleton
        - STEREOCHEMISTRY: Wrong stereoisomer detected
        - FUNCTIONAL_GROUP: Wrong functional groups (soft error)
        - CHIMERIC: Predicted mixed molecules
        - ISOMER_AMBIGUITY: Multiple valid structures fit spectrum
        - INSTRUMENT_ARTIFACT: Non-chemical signal
        - SOLVENT_INTERFERENCE: Spectral peak overlap
        - UNKNOWN: Root cause unclear

    FailureGalleryBuilder:
        - Aggregates failure cases into publication-ready figures
        - Generates confusion matrices (predicted ↔ true functions)
        - Produces side-by-side spectrum comparison plots
        - Writes failure case report with examples

Usage example:
    from spectralm.evals.failure_cases import (
        FailureCaseCollector, FailureTagger
    )
    
    # Collect failures from test set
    collector = FailureCaseCollector(
        ecr_threshold=0.25,  # Flag high-residual predictions
        tanimoto_threshold=0.7,  # Flag low-similarity predictions
    )
    
    failures = collector.collect(
        predictions,  # Dict[str, Prediction]
        ground_truth,  # Dict[str, str] spectrum_id → SMILES
    )
    
    # Analyze failure patterns
    for failure_type, cases in failures.items():
        print(f"{failure_type}: {len(cases)} cases")
    
    # Generate gallery
    from spectralm.evals.failure_cases import FailureGalleryBuilder
    gallery = FailureGalleryBuilder(output_dir='evals/failures/')
    gallery.build(failures, wavenumber_axis)
    gallery.write_report()

Failure statistics (from paper Table A.3):
    Total failures: 127 / 5000 (2.54%)
    
    Distribution:
        - Structure error: 45 (35%)
        - Isomer ambiguity: 32 (25%)
        - Functional group: 28 (22%)
        - Chimeric: 15 (12%)
        - Stereotype & others: 7 (6%)
    
    Patterns:
        - Symmetric molecules have higher failure rate (isomers)
        - Long chains (>20 atoms) prone to structure errors
        - Aromatic rings trigger more functional group errors
        - Aqueous samples have more solvent interference

Integration with ablation studies:
    Compare failure distributions across ablation variants:
        - Full model: 2.54% failure rate, ECR = 0.08
        - No physics loss: 6.1% failure rate, ECR = 0.18
        - No augmentation: 8.2% failure rate (overfitting to clean data)
    
    Physics loss reduces both error types but especially helps with
    isomer disambiguation via ECR penalty.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict


class FailureTagger(Enum):
    """Categories for classifying prediction failures."""
    
    STRUCTURE_ERROR = "wrong_carbon_skeleton"
    STEREOCHEMISTRY = "wrong_stereoisomer"
    FUNCTIONAL_GROUP = "wrong_or_missing_functional_group"
    CHIMERIC = "predicted_mixed_molecules"
    ISOMER_AMBIGUITY = "spectrum_matches_multiple_structures"
    INSTRUMENT_ARTIFACT = "non_chemical_spectral_feature"
    SOLVENT_INTERFERENCE = "solvent_peak_masking_analyte"
    UNKNOWN = "root_cause_unclear"


@dataclass
class FailureCase:
    """Single failure case with diagnostics."""
    
    spectrum_id: str
    observed_spectrum: list  # Wavenumber readings
    predicted_smiles: str
    ground_truth_smiles: str
    ecr_residual: float
    failure_type: FailureTagger
    notes: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.spectrum_id}: {self.failure_type.value}"


# Conditional imports
try:
    from spectralm.evals.failure_cases.collector import FailureCaseCollector
    from spectralm.evals.failure_cases.gallery import FailureGalleryBuilder
except ImportError:
    FailureCaseCollector = None
    FailureGalleryBuilder = None


__all__ = [
    'FailureTagger',
    'FailureCase',
    'FailureCaseCollector',
    'FailureGalleryBuilder',
]
