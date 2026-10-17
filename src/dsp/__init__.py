"""
src/dsp package
Member B Digital Signal Processing and Pulse Rate Variability engine.
"""

from .filters import filter_scg_cardiac, filter_scg_respiration, filter_ppg_cardiac
from .peak_detector import detect_systolic_peaks, compute_peak_intervals
from .prv_features import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    extract_poincare_descriptors
)
from .pipeline import process_cardiac_recording
from .quality import compute_signal_quality
from .synthetic_data import generate_synthetic_scg
from .batch_processor import generate_benchmark_training_corpus

__all__ = [
    "filter_scg_cardiac",
    "filter_scg_respiration",
    "filter_ppg_cardiac",
    "detect_systolic_peaks",
    "compute_peak_intervals",
    "extract_time_domain_features",
    "extract_frequency_domain_features",
    "extract_poincare_descriptors",
    "process_cardiac_recording",
    "compute_signal_quality",
    "generate_synthetic_scg",
    "generate_benchmark_training_corpus"
]
