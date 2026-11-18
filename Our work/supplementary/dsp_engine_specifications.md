> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Supplementary Technical Spec: Member 1 Core Sternal SCG DSP Engine
> **Correlating Template Section**: Precursor implementation & technical validation for *Chapter 3 (Methodology: Signal Conditioning & Feature Extraction)* and *Chapter 4 (Results)*.

## 1. Overview
This module contains the verified, zero-dependency Python digital signal processing (DSP) implementation for smartphone sternal Seismocardiography (SCG), owned by **Member 1 (Mobile Sternal SCG Acquisition & Edge DSP Lead)**.

## 2. Core Modules in `src/dsp/`
- **[`src/dsp/filters.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/dsp/filters.py)**:
  - 4th-order zero-phase Butterworth bandpass filter ($0.5\text{--}40.0\text{ Hz}$) for sternal SCG cardiac micro-vibrations.
  - Respiration bandpass filter ($0.1\text{--}0.5\text{ Hz}$) for baseline respiratory wander isolation.
- **[`src/dsp/quality.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/dsp/quality.py)**:
  - Absolute Physiological Energy Gate: verifies baseline dorso-ventral vibration standard deviation satisfies $0.005\text{ m/s}^2 \le \sigma_{raw} \le 0.50\text{ m/s}^2$. Rejects stationary tabletop vibrations ($<0.005\text{ m/s}^2$) or vigorous subject locomotion ($>0.50\text{ m/s}^2$) and triggers real-time stillness guidance before peak detection.
  - Quantitative Signal-to-Noise Ratio ($\text{SNR} \ge 12.0\text{ dB}$) and multi-parameter Signal Quality Index (SQI).
- **[`src/dsp/peak_detector.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/dsp/peak_detector.py)**:
  - Moving-RMS unipolar energy envelope calculation ($200\text{ ms}$ window, $K = \lfloor 0.2 \cdot f_s \rfloor$).
  - Dynamic sliding threshold $\theta[n] = \mu_E[n] + 0.60\sigma_E[n]$.
  - Physiological refractory window ($350\text{ ms}$) and interval clamping ($333\text{--}1500\text{ ms}$).
- **[`src/dsp/prv_features.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/dsp/prv_features.py)**:
  - Time-domain metrics: Mean HR, SDNN, RMSSD, pNN50.
  - Frequency-domain metrics: Cubic-spline resampling ($4.0\text{ Hz}$), Welch FFT PSD, VLF, LF ($0.04\text{--}0.15\text{ Hz}$), HF ($0.15\text{--}0.40\text{ Hz}$), and LF/HF sympathovagal ratio.
- **[`src/dsp/pipeline.py`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/src/dsp/pipeline.py)**:
  - Master pipeline entry point `process_cardiac_recording()` returning structured JSON feature vectors for Member 2's classification and simulation tier.
