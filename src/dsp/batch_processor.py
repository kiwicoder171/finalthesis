"""
src/dsp/batch_processor.py
Batch Dataset Feature Extraction Tool for Member C's Machine Learning.
Automates scanning directories of physiological recordings (CSV, text, or synthetic),
executing Member B's DSP pipeline, and exporting a clean training dataset CSV.
"""

import os
import csv
import json
from .pipeline import process_cardiac_recording
from .quality import compute_signal_quality
from .synthetic_data import generate_synthetic_scg

def generate_benchmark_training_corpus(output_csv="data/training_features_corpus.csv", n_subjects=60):
    """
    Generates a multi-subject benchmark training corpus across 3 classes:
    0: Normal Sinus Rhythm (Resting)
    1: Sympathetic Stress (Tachycardia + Low RMSSD + High LF/HF)
    2: Cardiac Arrhythmia / Ectopic Beats
    
    Exports a clean tabular dataset ready for Member C's scikit-learn models.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    rows = []
    headers = [
        "subject_id",
        "condition",
        "mean_hr_bpm",
        "mean_ppi_ms",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50_pct",
        "vlf_power",
        "lf_power",
        "hf_power",
        "lf_hf_ratio",
        "lf_nu",
        "hf_nu",
        "sd1",
        "sd2",
        "sd_ratio",
        "sqi_score",
        "stress_label",
        "arrhythmia_label"
    ]
    
    conditions = ["normal", "stressed", "arrhythmic"]
    
    for i in range(n_subjects):
        cond = conditions[i % 3]
        subj_id = f"SUBJ_{i+1:03d}_{cond.upper()}"
        base_rate = 65.0 + (i % 25)
        
        t, az, label_info = generate_synthetic_scg(
            duration_sec=30.0,
            fs=100.0,
            base_hr=base_rate,
            stress_level=cond,
            seed=1000 + i
        )
        
        # Run Member B DSP
        dsp_res = process_cardiac_recording(t, az, fs=100.0, mode="scg")
        sqi = compute_signal_quality(az, fs=100.0)
        
        f = dsp_res.get("feature_vector", {})
        
        row = [
            subj_id,
            cond,
            f.get("mean_hr_bpm", 0.0),
            f.get("mean_ppi_ms", 0.0),
            f.get("sdnn_ms", 0.0),
            f.get("rmssd_ms", 0.0),
            f.get("pnn50_pct", 0.0),
            f.get("vlf_power", 0.0),
            f.get("lf_power", 0.0),
            f.get("hf_power", 0.0),
            f.get("lf_hf_ratio", 0.0),
            f.get("lf_nu", 0.0),
            f.get("hf_nu", 0.0),
            f.get("sd1", 0.0),
            f.get("sd2", 0.0),
            f.get("sd_ratio", 0.0),
            sqi.get("sqi_score", 0.0),
            label_info.get("stress_class", 0),
            1 if label_info.get("arrhythmia", False) else 0
        ]
        rows.append(row)
        
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"[SUCCESS] Extracted {len(rows)} training rows into {output_csv}")
    return output_csv

if __name__ == "__main__":
    generate_benchmark_training_corpus()
