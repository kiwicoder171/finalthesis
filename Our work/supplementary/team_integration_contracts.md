> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Supplementary Technical Spec: Two-Member Team Integration Contracts
> **Correlating Template Section**: Non-template engineering contract (Inter-member API boundary for our two-person Capstone Project execution: Member 1 and Member 2).

## 1. Two-Member Responsibility Allocation
- **Member 1 (Mobile Sternal SCG Acquisition & Edge Digital Signal Processing — Tiers 1 & 2)**: Captures sternal SCG 3-axis MEMS accelerometer telemetry ($100\text{--}200\text{ Hz}$), enforces posture verification and the absolute physiological energy gate ($0.005 \le \sigma_{raw} \le 0.50\text{ m/s}^2$) with real-time stillness guidance, applies zero-phase 4th-order Butterworth bandpass filtering ($0.5\text{--}40.0\text{ Hz}$), detects aortic opening (AO) systolic peaks via moving-RMS adaptive thresholding, and computes time- and frequency-domain PRV metrics.
- **Member 2 (Machine Learning Arrhythmia Classification & In-Silico Digital Twin Simulation — Tiers 3 & 4)**: Ingests Member 1's extracted PRV feature vectors and clinical benchmark datasets (PhysioNet MIT-BIH & WESAD) to train and evaluate supervised ML classifiers (Decision Tree, Random Forest, Logistic Regression, and XGBoost) for Normal vs. Heart Problem / Arrhythmia and autonomic stress detection, while driving the coupled nonlinear Van der Pol ODE oscillator system (RK45) to visualize dynamic rhythm stability margins.

## 2. Standardized JSON Handoff Contract (Member 1 $\rightarrow$ Member 2)
Located at [`src/contracts/team_integration_guide.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/contracts/team_integration_guide.py):
```json
{
  "status": "SUCCESS",
  "modality": "STERNAL_SCG_MEMS",
  "feature_vector": {
    "mean_hr_bpm": 74.2,
    "mean_ppi_ms": 808.6,
    "sdnn_ms": 48.3,
    "rmssd_ms": 42.1,
    "pnn50_pct": 21.5,
    "lf_power": 412.5,
    "hf_power": 328.1,
    "lf_hf_ratio": 1.257,
    "sqi_score": 1.0,
    "quality_grade": "OPTIMAL",
    "is_usable": true
  },
  "diagnostics": {
    "sample_count": 3737,
    "duration_sec": 37.5,
    "sampling_rate_hz": 99.5,
    "raw_std_ms2": 0.0669,
    "detected_beat_count": 46
  }
}
```
