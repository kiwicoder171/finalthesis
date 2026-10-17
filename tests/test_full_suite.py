#!/usr/bin/env python3
"""
tests/test_full_suite.py
Comprehensive test suite validating Member B's complete DSP & Feature Engine.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dsp.filters import design_butterworth_bandpass, apply_zero_phase_filter, filter_scg_cardiac, filter_ppg_cardiac
from src.dsp.peak_detector import detect_systolic_peaks, compute_peak_intervals
from src.dsp.prv_features import extract_time_domain_features, extract_frequency_domain_features, extract_poincare_descriptors
from src.dsp.quality import compute_signal_quality
from src.dsp.synthetic_data import generate_synthetic_scg
from src.dsp.pipeline import process_cardiac_recording
from src.contracts.team_integration_guide import member_c_quickstart_example, member_d_quickstart_example

def test_filters():
    print("[TEST 1/6] Testing Butterworth Bandpass Filters...")
    b, a = design_butterworth_bandpass(1.5, 15.0, 100.0)
    assert len(b) == 3 and len(a) == 3, "Filter coefficients dimension mismatch"
    dummy = [1.0] * 50
    filtered = apply_zero_phase_filter(b, a, dummy)
    assert len(filtered) == 50, "Filtered length mismatch"
    print("  -> Passed.")

def test_peaks_and_intervals():
    print("[TEST 2/6] Testing Systolic Peak Detector...")
    # Synthetic 1 Hz sine (60 bpm)
    fs = 100.0
    synth = [math.sin(2.0 * math.pi * 1.0 * (i / fs)) for i in range(300)]
    peaks = detect_systolic_peaks(synth, fs)
    assert len(peaks) >= 2, "Failed to detect synthetic periodic peaks"
    raw_ppi, clean_ppi, _ = compute_peak_intervals(peaks, [i / fs for i in range(300)])
    assert len(clean_ppi) >= 1, "Failed to compute clean intervals"
    print("  -> Passed.")

def test_prv_metrics():
    print("[TEST 3/6] Testing Time and Frequency PRV Features...")
    test_ppi = [800.0, 810.0, 795.0, 805.0, 820.0, 790.0, 815.0, 800.0, 805.0, 810.0]
    time_feat = extract_time_domain_features(test_ppi)
    assert 70.0 < time_feat["mean_hr_bpm"] < 80.0, "Heart rate calculation mismatch"
    assert time_feat["sdnn_ms"] > 0, "SDNN must be positive"
    
    freq_feat = extract_frequency_domain_features(test_ppi)
    assert "lf_hf_ratio" in freq_feat, "Missing LF/HF ratio in spectral output"
    
    poincare = extract_poincare_descriptors(test_ppi)
    assert "sd1" in poincare and "sd2" in poincare, "Missing Poincaré descriptors"
    print("  -> Passed.")

def test_quality_and_motion():
    print("[TEST 4/6] Testing Signal Quality Index (SQI)...")
    _, clean_scg, _ = generate_synthetic_scg(duration_sec=10.0, fs=100.0)
    sqi = compute_signal_quality(clean_scg, 100.0)
    assert sqi["quality_grade"] in ["OPTIMAL", "ACCEPTABLE"], f"Clean SCG misclassified: {sqi}"
    
    flatline = [0.0] * 500
    sqi_bad = compute_signal_quality(flatline, 100.0)
    assert sqi_bad["quality_grade"] == "NOISY_REJECT", "Flatline not rejected"
    print("  -> Passed.")

def test_pipeline_real_and_synthetic():
    print("[TEST 5/6] Testing Pipeline Execution...")
    t, az, _ = generate_synthetic_scg(duration_sec=15.0, fs=100.0, base_hr=72.0)
    res = process_cardiac_recording(t, az, mode="scg")
    assert res["success"] is True, "Pipeline failed on synthetic data"
    assert "feature_vector" in res, "Missing feature vector"
    print("  -> Passed.")

def test_contracts():
    print("[TEST 6/6] Testing Team Integration Contracts...")
    features = member_c_quickstart_example()
    assert "mean_hr_bpm" in features and "lf_hf_ratio" in features
    params = member_d_quickstart_example(features, risk_score=0.75)
    assert "omega_0" in params and "mu" in params
    print("  -> Passed.")

if __name__ == "__main__":
    import math
    print("=" * 60)
    print("MEMBER B COMPREHENSIVE VERIFICATION TEST SUITE")
    print("=" * 60)
    test_filters()
    test_peaks_and_intervals()
    test_prv_metrics()
    test_quality_and_motion()
    test_pipeline_real_and_synthetic()
    test_contracts()
    print("=" * 60)
    print("ALL 6 TEST MODULES PASSED WITH 100% SUCCESS!")
    print("=" * 60)
