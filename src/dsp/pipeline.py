"""
src/dsp/pipeline.py
The unified Digital Signal Processing (DSP) pipeline for Member B.
Provides a clean, one-call interface for processing raw cardiac time-series
and outputting clinical biomarkers ready for Member C's ML risk model.
"""

from .filters import filter_scg_cardiac, filter_scg_respiration, filter_ppg_cardiac
from .peak_detector import detect_systolic_peaks, compute_peak_intervals
from .prv_features import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    extract_poincare_descriptors
)
from .quality import compute_signal_quality

def process_cardiac_recording(timestamps, signal, fs=None, mode="scg"):
    """
    Executes the complete Member B DSP & PRV extraction pipeline.
    
    Args:
        timestamps (list): 1D array of timestamps in seconds.
        signal (list): 1D array of raw sensor measurements (a_z or optical green channel).
        fs (float, optional): Sampling frequency. If None, computed from timestamps.
        mode (str): Sensing modality: 'scg' (chest accelerometer) or 'ppg' (camera pulse).
        
    Returns:
        dict: Complete analysis packet with filtered waveforms and feature metrics.
    """
    n_samples = len(signal)
    if n_samples < 20:
        return {"error": "Insufficient data length"}
        
    # 1. Compute Sampling Frequency if not provided
    if fs is None:
        diffs = [timestamps[i] - timestamps[i-1] for i in range(1, n_samples)]
        mean_dt = sum(diffs) / len(diffs)
        fs = 1.0 / mean_dt if mean_dt > 0 else 100.0
        
    # 2. Baseline Wander & DC Gravity Removal
    mean_val = sum(signal) / n_samples
    ac_signal = [x - mean_val for x in signal]
    
    # 3. Multi-Modal Digital Butterworth Filtering
    if mode.lower() == "ppg":
        filtered_cardiac = filter_ppg_cardiac(ac_signal, fs)
        filtered_resp = []
    else: # Default SCG
        filtered_cardiac = filter_scg_cardiac(ac_signal, fs)
        filtered_resp = filter_scg_respiration(ac_signal, fs)
        
    # 4. Adaptive Systolic Peak Detection
    peak_indices = detect_systolic_peaks(filtered_cardiac, fs)
    raw_ppi, clean_ppi, clean_times = compute_peak_intervals(peak_indices, timestamps)
    
    # 5. Clinical Feature Extraction
    time_metrics = extract_time_domain_features(clean_ppi)
    freq_metrics = extract_frequency_domain_features(clean_ppi)
    poincare_metrics = extract_poincare_descriptors(clean_ppi)
    quality_metrics = compute_signal_quality(signal, fs)
    
    # 6. Combined Feature Vector (Strict Handoff to Member C)
    feature_vector = {
        **time_metrics,
        **freq_metrics,
        **poincare_metrics,
        **quality_metrics,
        "sample_count": n_samples,
        "duration_sec": round(timestamps[-1] - timestamps[0], 2),
        "sampling_rate_hz": round(fs, 2),
        "detected_beat_count": len(clean_times),
        "modality": mode.upper()
    }
    
    return {
        "success": True,
        "fs": fs,
        "duration": timestamps[-1] - timestamps[0],
        "filtered_cardiac": filtered_cardiac,
        "filtered_resp": filtered_resp,
        "peak_indices": peak_indices,
        "clean_ppi_ms": clean_ppi,
        "feature_vector": feature_vector
    }
