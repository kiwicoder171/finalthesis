"""
Standardized ECG Benchmarking Package (Contract 3).
"""

from src.benchmark.metrics import AAMI_CLASSES, compute_confusion_matrix, evaluate_ec57_metrics
from src.benchmark.report_generator import generate_benchmark_report
from src.benchmark.runner import ECGBenchmarkRunner, detect_r_peaks_and_annotate

__all__ = [
    "AAMI_CLASSES",
    "compute_confusion_matrix",
    "evaluate_ec57_metrics",
    "ECGBenchmarkRunner",
    "detect_r_peaks_and_annotate",
    "generate_benchmark_report",
]
