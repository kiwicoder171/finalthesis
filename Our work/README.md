> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Our Work — Master Organization & Template Mapping Index

This directory organizes all capstone thesis materials for our **two-member engineering team**, unifying the methodology around **100% Smartphone Sternal Seismocardiography (SCG)**, incorporating advisor feedback, and directly mapping our work to the official Mahidol University **EGCO Bachelor Thesis Template 2023**.

---

## 1. Thesis Chapters (Correlated Directly to Template)

All five chapters are fully formatted in publication-grade LaTeX and synchronized with `00 EGCO Bachelor Thesis Template 2023 for Upload/`:

| File | Template Target | Description & Advisor Feedback Incorporated |
| :--- | :--- | :--- |
| **[`chapter1.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter1.tex)** | `chapter1.tex` (Introduction) | • **Unified Method**: Aligned 100% to Smartphone Sternal MEMS SCG mechanical sensing (zero external hardware, zero optical PPG).<br>• **Advisor Comments**: Added explicit performance metrics (Accuracy $\ge 85\%$, Sensitivity $\ge 88\%$, Specificity $\ge 85\%$, F1 $\ge 0.86$, MAE $<3$ bpm) and explicit binary classification: *Normal Sinus Rhythm* vs. *Heart Problem / Arrhythmia (PVCs, PACs, dysrhythmias)*.<br>• Formatted with action verbs in `\section{Objective}`, detailed 5-point `\section{Scope}`, full `\section{Timeline}`, and `\section{Division of Responsibilities}` for the **two-member team** (Member 1: Mobile Sternal SCG & Edge DSP; Member 2: ML Arrhythmia Classification & In-Silico Digital Twin). |
| **[`chapter2.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter2.tex)** | `chapter2.tex` (Literature Review) | • **Advisor Comments**: Researched 5 directly related mobile cardiac & processing projects (Lee et al., Quiroz-Juárez et al., Tsamis et al., Chai, Aliev et al.).<br>• For **every single paper**, explicitly details: (1) Proposed method, (2) Data used, (3) Performance metrics evaluated, (4) Key findings, (5) **What we can steal (adopt)**, and (6) **How our work is different / our competitive edge**.<br>• Includes a comprehensive summary comparison matrix table (`Table 2.1`). |
| **[`chapter3.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter3.tex)** | `chapter3.tex` (Methodology) | • **Reworked per Advisor Feedback (Single Abnormality: Arrhythmia Only)** matching all 5 template sections:<br>&nbsp;&nbsp;1. `\section{System Design}`: 3-stage achievable workflow (Heartbeat Acquisition $\to$ 2-Person Comparison $\to$ Arrhythmia Screening), 4-tier SCG architecture diagram, Smartphone SCG vs. Smartwatch rationale, decision flowchart, and `Algorithm 1`.<br>&nbsp;&nbsp;2. `\section{Data}`: Optimal placement protocol (lower sternum / xiphoid, supine $Z$-axis + site comparison), posture/energy gating, and 3-tier achievable/explainable data strategy (Empirical Smartphone SCG + PhysioNet MIT-BIH + Controlled Synthetic Arrhythmic SCG Simulation; WESAD/stress removed).<br>&nbsp;&nbsp;3. `\section{Methods}`: 4th-order zero-phase Butterworth filter ($0.5\text{--}40.0\text{ Hz}$), moving-RMS envelope, adaptive AO peak detection, explainable PRV features, Stage 2 inter-subject differentiation, Stage 3 explainable binary arrhythmia classifiers (LR, DT, RF, XGBoost), and Van der Pol perturbation simulation.<br>&nbsp;&nbsp;4. `\section{Hyperparameter Configurations}`: Complete table of filter, windowing, and energy-gating ($0.005\text{--}0.50\text{ m/s}^2$) parameters.<br>&nbsp;&nbsp;5. `\section{Performance Evaluation}`: Mobile-phone evaluation protocol (simultaneous reference MAE/RMSE/Bland--Altman, placement/gating bench tests, 2-person comparison), binary arrhythmia metrics (Acc, Sens, Spec, Prec, F1), patient-stratified 5-fold CV, and explainability audit. |
| **[`ADVISOR_MEETING_QA_AND_ACTION_PLAN.md`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/ADVISOR_MEETING_QA_AND_ACTION_PLAN.md)** | Advisor Defense Prep | • Concise defense Q&A answering all committee questions from the advisor meeting (Smartwatch vs. SCG, optimal SCG placement, mobile evaluation protocol, core motivation) plus a data-first 3-stage execution checklist. |
| **[`chapter4.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter4.tex)** | `chapter4.tex` (Result) | • **Empirical & Clinical Benchmark Validation**:<br>&nbsp;&nbsp;1. Physiological energy gating validation ($100\%$ stationary tabletop rejection at $\sigma_{raw}=0.0018\text{ m/s}^2$, motion rejection at $0.842\text{ m/s}^2$, valid resting sternal SCG at $0.0669\text{ m/s}^2$ with $+14.8\text{ dB}$ SNR gain).<br>&nbsp;&nbsp;2. Systolic peak detection precision ($\text{MAE} = 2.14\text{ bpm}$, $\text{RMSE} = 3.28\text{ bpm}$, Bland--Altman bias $+0.32\text{ bpm}$, LoA $[-4.18, +4.82]\text{ bpm}$).<br>&nbsp;&nbsp;3. Supervised ML arrhythmia classification on MIT-BIH (XGBoost: $89.4\%$ Accuracy, $88.9\%$ Sensitivity, $89.8\%$ Specificity, $0.887$ F1-score) and WESAD stress detection ($86.2\%$ Accuracy).<br>&nbsp;&nbsp;4. Coupled Van der Pol RK45 limit-cycle simulation and soft real-time latency breakdown ($56.1\text{ ms}$ total, well below the $500\text{ ms}$ ceiling). |
| **[`chapter5.tex`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/chapter5.tex)** | `chapter5.tex` (Conclusion) | • **Synthesis, Obstacles & Future Roadmap**:<br>&nbsp;&nbsp;1. `\section{Conclusion}`: Summary of technical accomplishments across Tiers 1--4.<br>&nbsp;&nbsp;2. `\section{Obstacles and Limitations}`: Posture dependence of sternal SCG, consumer MEMS sensor noise floors, physical sternal coupling variability, and mobile OS timestamp jitter.<br>&nbsp;&nbsp;3. `\section{Future Work}`: On-device TinyML NPU quantization, 6-DoF multi-sensor inertial gyroscope fusion (GCG), bi-directional digital-twin closed loop, and multi-center clinical validation. |
| **[`references.bib`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/references.bib)** | `references.bib` | Clean, unified BibTeX bibliography containing all 25 cited peer-reviewed papers with 100% 1-to-1 citation parity (0 missing, 0 uncited). |

---

## 2. Historical Raw Drafts (Preserved as Backups)

Located in **[`raw_drafts/`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/raw_drafts)**:
- `chapter1_finished.tex`: Original early Chapter 1 draft (before unifying around pure smartphone sternal SCG).
- `chapter2_finished.tex`: Original early Chapter 2 draft (before adding the 5 peer reviews and "steal/different" sections).
- `chapter3_notes.txt`: Original early brainstorm notes for Chapter 3.

---

## 3. Supplementary Engineering Work (Two-Member Team Specs)

These components document our completed software, interface contracts, and empirical benchmarks supporting the thesis:

| Subfolder / Document | Scope & Purpose |
| :--- | :--- |
| **[`supplementary/dsp_engine_specifications.md`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/supplementary/dsp_engine_specifications.md)** | Technical specification for **Member 1's Core Sternal SCG DSP Engine** (`src/dsp/`): 4th-order zero-phase Butterworth filter ($0.5\text{--}40.0\text{ Hz}$), absolute physiological energy gate ($0.005\text{--}0.50\text{ m/s}^2$), adaptive peak detector, and PRV feature extractor. |
| **[`supplementary/team_integration_contracts.md`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/supplementary/team_integration_contracts.md)** | Two-member interface contract defining the exact JSON handoff schema between **Member 1** (Mobile Sternal SCG Acquisition & Edge DSP, Tiers 1--2) and **Member 2** (Machine Learning Arrhythmia Classification & In-Silico Digital Twin Simulation, Tiers 3--4). |
| **[`supplementary/empirical_sensor_benchmarks.md`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/supplementary/empirical_sensor_benchmarks.md)** | Hardware validation benchmarks from a real iPhone sternal SCG recording (`data/raw_phyphox_test/Raw Data.csv`), showing effective $99.52\text{ Hz}$ sampling, $0.0669\text{ m/s}^2$ vibration standard deviation, and successful beat extraction. |
| **Interactive Web Prototypes** | Live prototypes located in `public/`: `scg_inspection_dashboard.html`, `mems_scg_anatomy.html`, and `public/static/` (PWA mobile sternal SCG recorder and desktop analytical dashboard). |

---

## 4. Academic Citation Flow Protocol

See **[`ACADEMIC_CITATION_FLOW.md`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/Our%20work/ACADEMIC_CITATION_FLOW.md)** for the 5-step procedure used to verify all 25 primary academic sources across Chapters 1--5.
