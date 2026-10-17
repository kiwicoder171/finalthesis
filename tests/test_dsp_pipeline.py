#!/usr/bin/env python3
"""
tests/test_dsp_pipeline.py
Unit tests verifying Member B's complete DSP pipeline using the real user recording.
"""

import sys
import os
import csv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dsp import process_cardiac_recording

def run_tests():
    csv_path = "data/raw_phyphox_test/Raw Data.csv"
    if not os.path.exists(csv_path):
        print(f"Test data not found at {csv_path}")
        return False
        
    timestamps = []
    az_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                try:
                    timestamps.append(float(row[0]))
                    az_data.append(float(row[3]))
                except ValueError:
                    continue
                    
    print(f"Loaded {len(timestamps)} samples from real iPhone recording.")
    
    # Run Member B's complete pipeline
    result = process_cardiac_recording(timestamps, az_data, mode="scg")
    
    assert result["success"] is True, "Pipeline execution failed!"
    features = result["feature_vector"]
    
    print("\n--- MEMBER B EXTRACTED FEATURE VECTOR (HANDOFF TO MEMBER C) ---")
    for key, val in features.items():
        print(f"  {key:20s}: {val}")
        
    assert features["mean_hr_bpm"] > 50 and features["mean_hr_bpm"] < 130, f"Heart rate out of physiological range: {features['mean_hr_bpm']}"
    assert features["detected_beat_count"] > 20, f"Insufficient beats detected: {features['detected_beat_count']}"
    assert features["sdnn_ms"] > 0, "SDNN should be positive"
    assert features["rmssd_ms"] > 0, "RMSSD should be positive"
    assert features["lf_hf_ratio"] > 0, "LF/HF ratio should be positive"
    
    print("\n[PASSED] All DSP pipeline tests succeeded!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
