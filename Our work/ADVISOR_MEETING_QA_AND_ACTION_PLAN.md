> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Advisor Meeting Notes — Rework Summary & Defense Q&A Guide

This document directly addresses every comment, suggestion, and potential committee question raised in the advisor meeting, aligned with the reworked [`chapter3.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter3.tex).

---

## 1. What Was Fixed in `Our work/chapter3.tex`

1. **Focused Strictly on 1 Abnormality (`Cardiac Arrhythmia`):**
   - Removed the secondary Autonomic Stress / WESAD classification task from Chapter 3 so the entire methodology is laser-focused on **Normal Sinus Rhythm vs. Cardiac Arrhythmia**.
2. **Restructured Around the Advisor's 3-Stage Achievable Workflow:**
   - **Stage 1 (Acquisition):** Heartbeat detection on a smartphone via lower-sternal SCG ($Z$-axis MEMS accelerometer), zero-phase Butterworth filtering ($0.5\text{--}40\text{ Hz}$), energy gating ($0.005 \le \sigma_{raw} \le 0.50\text{ m/s}^2$), and adaptive AO peak detection.
   - **Stage 2 (Inter-Subject Comparison):** Differentiate between 2+ people using empirical phone recordings and public data to prove the pipeline captures real individual physiology rather than sensor noise.
   - **Stage 3 (Arrhythmia Detection):** Binary classification (*Normal* vs. *Arrhythmia*) using **explainable** PRV features (`Mean HR`, `SDNN`, `RMSSD`, `pNN50`, `LF`, `HF`, `LF/HF`) on public **PhysioNet MIT-BIH** data plus **controlled synthetic arrhythmic SCG simulation** (real sternal AO templates modulated by MIT-BIH arrhythmic intervals / Van der Pol perturbations).
3. **Added Optimal SCG Placement & Comparison Protocol** (`Section 3.2.1`).
4. **Added Smartphone-Side Evaluation Protocol** (`Section 3.5.1`) and **Smartwatch vs. SCG Engineering Rationale** (`Section 3.1.2`).

---

## 2. Defense Q&A — Answers to Every Potential Committee Question

### Q1: Is Smartphone SCG better than a Smartwatch? What can't a watch do compared to SCG, and how can we achieve that?

**Direct, honest academic framing (don't overclaim):**
A smartwatch is better for *continuous 24/7 tracking during walking/exercise*. However, smartphone sternal SCG has **four concrete advantages** over a smartwatch for **on-demand resting cardiac screening**:

| Dimension | Smartwatch (Wrist PPG / 1-Lead ECG) | Smartphone Sternal SCG (Our Approach) |
| :--- | :--- | :--- |
| **Hardware Cost** | Requires buying a \$200–\$800+ dedicated wearable | **\$0 incremental hardware** — uses the smartphone the user already owns |
| **What It Measures** | Peripheral optical blood volume pulse (PPG) at the wrist (or 1-lead electrical ECG on premium models) | **Direct central cardiac mechanics** — ventricular ejection recoil & aortic valve opening (AO) right over the heart |
| **Signal Biases & Lag** | Affected by skin pigmentation, cold/poor peripheral circulation, wrist-strap tightness, and vascular Pulse Transit Time (PTT) jitter | **Zero optical skin-tone bias** and **zero peripheral PTT lag** (mechanical recoil travels straight through the sternum) |
| **Transparency / Explainability** | Closed proprietary algorithms ("Inconclusive" / black-box alerts) | **100% explainable DSP + PRV features** (`Mean HR`, `SDNN`, `RMSSD`, `pNN50`, `LF/HF`) with explicit quality gating (`SNR >= 12 dB`) |

**What a watch *cannot* do compared to SCG:**
- A wrist watch **cannot capture central chest-wall mechanical cardiac events** (aortic valve opening/closing, myocardial recoil energy, or electromechanical coupling). Even an ECG watch only sees electrical depolarization (Lead I), not whether the heart mechanically ejected blood effectively.
- A wrist PPG watch **cannot eliminate optical skin-tone/perfusion artifacts or peripheral PTT delay**.

**How we achieve this advantage in our project:**
1. Keep the system **zero-hardware** (standard iOS/Android MEMS accelerometer).
2. Enforce **deterministic physiological quality gating** ($\sigma_{raw} \in [0.005, 0.50]\text{ m/s}^2$, $\text{SNR} \ge 12\text{ dB}$) so bad readings are rejected instead of producing fake heart rates.
3. Expose **explainable PRV & beat-irregularity features** rather than an opaque black-box score.

---

### Q2: Which is the best position for SCG to work best?

**Answer:** **Lower third of the sternum (xiphoid process / lower sternal body)**, with the smartphone lying **flat on the chest** while the user is **supine (lying flat on their back)**, measuring the **dorso-ventral ($Z$) axis**.

**Why this position is biomechanically best:**
1. **Rigid Bony Coupling:** The sternum sits directly over the anterior heart wall and transmits cardiac micro-vibrations cleanly with minimal soft-tissue damping (unlike lateral ribs or soft abdominal tissue).
2. **Direct Axis Alignment:** Lying supine aligns the smartphone's $Z$-axis perpendicular to ventricular ejection recoil and aortic valve opening (AO).
3. **Gravity-Stabilized Contact (Zero Hand Tremor):** Gravity holds the phone flat against the chest without the user gripping it, eliminating hand-holding muscle tremor.

**How to prove it experimentally (10-minute test):**
Record 30–60s windows at 4 positions on the same person and report a small table of $\sigma_{raw}$ and post-filter SNR ($\text{dB}$):
1. Lower sternum (xiphoid), supine *(expected highest SNR)*
2. Upper sternum (manubrium), supine
3. Left 4th–5th intercostal space (cardiac apex), supine
4. Seated upright holding the phone against the chest *(expected lowest SNR due to hand tremor & gravity tilt)*

---

### Q3: How can we evaluate from a mobile phone what we find?

Because we cannot easily recruit clinical arrhythmia patients for live phone testing in an undergraduate project, evaluation is split into **three clean, verifiable layers**:

1. **Layer 1 — Simultaneous Reference Validation on Real Phone Data (Stage 1):**
   - Record resting supine sternal SCG on the smartphone while simultaneously wearing an accessible reference (fingertip pulse oximeter, smartwatch/chest-strap ECG, or clinical ECG).
   - Quantify agreement using **MAE ($< 3.0\text{ bpm}$)**, **RMSE ($< 4.5\text{ bpm}$)**, and **Bland–Altman limits of agreement**.
   - Run **bench rejection tests**: phone on a flat table ($100\%$ rejected, $\sigma_{raw} < 0.005\text{ m/s}^2$) and walking/talking ($100\%$ rejected, $\sigma_{raw} > 0.50\text{ m/s}^2$).
2. **Layer 2 — 2-Person Differentiation Check (Stage 2):**
   - Record multiple 60s windows from at least **2 different subjects** (e.g., Team Member 1 vs. Team Member 2) and show that extracted `Mean HR`, `SDNN`, `RMSSD`, and `pNN50` cluster consistently per person and match their reference devices.
3. **Layer 3 — Arrhythmia Detection on Public Clinical Data + Controlled SCG Simulation (Stage 3):**
   - Train and test the binary classifier (*Normal* vs. *Arrhythmia*) on **patient-stratified PhysioNet MIT-BIH** holdout sets (reporting **Accuracy, Sensitivity $\ge 88\%$, Specificity $\ge 85\%$, F1 $\ge 0.86$**, and Confusion Matrix).
   - Validate end-to-end phone DSP-to-ML behavior using **controlled synthetic arrhythmic SCG** (placing real healthy phone AO beat templates onto clinical MIT-BIH arrhythmic beat-interval sequences + realistic MEMS noise) to prove the smartphone pipeline still detects the irregularity.

---

### Q4: Why do we do this, and what's the best reason to do this?

**The strongest, most defensible answer:**
1. **Healthcare Accessibility & Screening Equity (Primary Reason):** Cardiovascular diseases cause ~17.9 million deaths per year globally, with the largest burden in low- and middle-income populations where clinical Holter monitors and \$300+ smartwatches are unaffordable or unavailable. Smartphones, however, are nearly universal. Turning a phone's built-in MEMS accelerometer into a **zero-hardware, on-demand arrhythmia pre-screening tool** removes the cost barrier to early cardiac checkups.
2. **Explainable, Non-Opaque Screening (Engineering Reason):** Commercial consumer wearables act as closed black boxes. Our pipeline provides **mathematically transparent DSP quality gating** and **clinically interpretable PRV features** (`RMSSD`, `SDNN`, `pNN50`), so every screening alert can be traced back to an actual physiological rhythm irregularity.

---

## 3. Step-by-Step Execution Checklist (Data-First)

- [ ] **Step 1 — Heartbeat Acquisition & Placement Test:** Capture 60s supine lower-sternal SCG recordings (plus upper sternum, apex, seated, tabletop, and motion controls) alongside a reference HR monitor; verify MAE $< 3.0\text{ bpm}$ and SNR $\ge 12\text{ dB}$.
- [ ] **Step 2 — 2-Person Comparison:** Collect 3–5 resting windows from Subject A and Subject B (and compare with 2 distinct PhysioNet MIT-BIH subjects) to demonstrate clear, repeatable inter-subject feature differentiation.
- [ ] **Step 3 — Single-Abnormality Arrhythmia Model + Simulation:** Train/evaluate the binary *Normal vs. Arrhythmia* classifiers (LR, DT, RF, XGBoost) on MIT-BIH with patient-stratified splits, inspect feature importances for explainability, and run the synthetic arrhythmic SCG stress test.
- [ ] **Step 4 — Sync Downstream Chapters (after advisor approval of Chapter 3):** Align `chapter1.tex`, `chapter4.tex`, `chapter5.tex`, and `00 EGCO Bachelor Thesis Template 2023 for Upload/` to match the single-abnormality Arrhythmia focus (removing WESAD/stress sections).
