> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# CardioSense (Smartphone Sternal SCG Screening)

Domain vocabulary for the zero-hardware smartphone sternal seismocardiography (SCG) cardiac screening thesis (`Our work/` Chapters 1–3).

## Language

### Sensing & Signal Quality

**Sternal Seismocardiography (Sternal SCG)**:
Non-invasive measurement of central cardiac mechanical micro-vibrations along the dorso-ventral ($Z$) axis using a smartphone resting flat over the lower sternum.
_Avoid_: Chest PPG, optical pulse sensing, wearable ECG

**Aortic Opening (AO) Peak**:
The dominant systolic mechanical peak in the sternal SCG waveform corresponding to aortic valve opening and ventricular ejection recoil.
_Avoid_: R-peak (reserve R-peak for ECG ground truth), heartbeat spike

**Peak-to-Peak Interval (PPI)**:
The elapsed time in milliseconds between consecutive verified AO peaks, serving as the mechanical surrogate for ECG R–R intervals.
_Avoid_: Heartbeat gap, raw RR interval (when derived from SCG)

**Physiological Energy Gate**:
Deterministic pre-filtering boundary ($0.005 \le \sigma_{raw} \le 0.50\text{ m/s}^2$) that rejects stationary inanimate surfaces and motion-corrupted buffers before peak detection.
_Avoid_: Noise filter, soft threshold

**Signal Quality Index (SQI)**:
Composite pass/fail validity status combining posture alignment ($\pm 10^\circ$), physiological energy bounds, and post-filter signal-to-noise ratio ($\text{SNR} \ge 12\text{ dB}$).
_Avoid_: Confidence score, quality rating

### Clinical Scope & Screening Taxonomy

**Rhythm-Disorder Cluster**:
The single-scope screening target comprising three cardiac abnormalities observable strictly from resting beat-to-beat PPI/PRV timing and heart rate: Ectopic Beats, Irregular Rhythm / Atrial Fibrillation, and Rate Disorders.
_Avoid_: Multi-disease diagnosis, stress classification, 12-lead morphology screening

**Ectopic Beat**:
A premature contraction arising outside the sinoatrial node—specifically Premature Ventricular Contractions (PVC, AAMI `V`) or Premature Atrial/Supraventricular Contractions (PAC, AAMI `S`).
_Avoid_: Extra beat, structural arrhythmia

**Irregular Rhythm / Atrial Fibrillation (AFib)**:
Sustained beat-to-beat interval irregularity characterized by elevated short-term variability (`RMSSD`, `SDNN`, `pNN50`) without regular sinus periodicity.
_Avoid_: General tachycardia, heart failure

**Rate Disorder**:
Resting heart-rate deviation outside normal adult sinus bounds, partitioned into Sinus Bradycardia ($< 60\text{ bpm}$) and Sinus Tachycardia ($> 100\text{ bpm}$).
_Avoid_: Exercise elevation, stress response

**Two-Stage Hybrid Screening**:
The screening architecture that flags Rate Disorders via deterministic physiological HR thresholds and classifies Rhythm Arrhythmias (Normal vs. Ectopic vs. Irregular/AFib, plus a Binary Abnormal Flag) via supervised machine learning.
_Avoid_: Black-box classifier, end-to-end deep diagnosis

### Validation, Benchmarking & Positioning

**Patient-Stratified Split**:
Dataset partitioning in which every recording and beat from a given patient belongs exclusively to either training or testing, preventing inter-patient data leakage.
_Avoid_: Random beat split, record-mixed split

**Complementary Dual-Mode Pathway**:
Optional future partnership framing in which a smartwatch provides continuous 24/7 ambulatory monitoring alerts while smartphone sternal SCG provides zero-cost, on-demand central mechanical confirmation at rest.
_Avoid_: Smartwatch replacement, wearable dependency

**System Usability Scale (SUS)**:
Standardized 10-item quantitative usability score ($0\text{--}100$, target $\ge 75$) paired with guided setup time ($< 60\text{ s}$) and first-attempt valid-capture rate ($\ge 90\%$) to measure user-friendliness.
_Avoid_: Subjective friendliness, qualitative UX
