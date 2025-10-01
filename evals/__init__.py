"""
evals/ — SpectraLM evaluation and analysis suite.

Public API:
    DomainResidualScorer  — full evaluation harness (domain_residuals.py)
    AblationRunner        — systematic ablation study (ablation_runner.py)
    SignificanceTester    — bootstrap CI + paired t-tests (ablation_runner.py)
    FailureCaseCollector  — mines high-ECR predictions (failure_cases/collector.py)
    FailureGalleryBuilder — renders annotated HTML gallery (failure_cases/gallery.py)

Quickstart:
    from evals import DomainResidualScorer
    scorer = DomainResidualScorer(model, device="cuda")
    report = scorer.evaluate(test_loader, test_smiles)
    report.print_summary()
    report.save("evals/results/my_run.json")

Failure gallery:
    from evals import FailureCaseCollector, FailureGalleryBuilder
    collector = FailureCaseCollector(model, device="cuda")
    cases     = collector.collect_from_loader(test_loader, test_smiles, top_n=50)
    collector.save_all(cases, "evals/failure_cases/annotated")
    FailureGalleryBuilder().build(cases, "evals/failure_cases")
"""

from evals.domain_residuals import DomainResidualScorer, EvalReport, SampleResult
from evals.ablation_runner import AblationRunner, AblationVariant, SignificanceTester
from evals.failure_cases.collector import FailureCaseCollector, FailureCase
from evals.failure_cases.gallery import FailureGalleryBuilder

__all__ = [
    "DomainResidualScorer",
    "EvalReport",
    "SampleResult",
    "AblationRunner",
    "AblationVariant",
    "SignificanceTester",
    "FailureCaseCollector",
    "FailureCase",
    "FailureGalleryBuilder",
]