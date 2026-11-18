"""
Data ingestion, cataloging, preprocessing, and segmentation pipeline for ECG analysis.
"""

from src.data.ecg_catalog import (
    ALL_MITBIH_RECORDS,
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_RECORDS,
    DatasetMetadata,
    ECGCatalog,
)
from src.data.preprocessor import ECGPreprocessor, resample_indices
from src.data.segmenter import (
    AAMI_BEAT_MAP,
    AAMI_CLASS_TO_INT,
    INT_TO_AAMI_CLASS,
    NON_BEAT_SYMBOLS,
    compute_rhythm_features,
    extract_heartbeats,
    is_qrs_beat,
    map_symbol_to_aami,
    normalize_annotations,
)
from src.data.wfdb_streamer import WFDBStreamer

__all__ = [
    "ALL_MITBIH_RECORDS",
    "DS1_RECORDS",
    "DS2_RECORDS",
    "PACED_RECORDS",
    "DatasetMetadata",
    "ECGCatalog",
    "ECGPreprocessor",
    "resample_indices",
    "AAMI_BEAT_MAP",
    "AAMI_CLASS_TO_INT",
    "INT_TO_AAMI_CLASS",
    "NON_BEAT_SYMBOLS",
    "compute_rhythm_features",
    "extract_heartbeats",
    "is_qrs_beat",
    "map_symbol_to_aami",
    "normalize_annotations",
    "WFDBStreamer",
]
