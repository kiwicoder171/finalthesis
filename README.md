# CardioSense: Zero-Hardware Cardiac Screening via Smartphone Sternal Seismocardiography

**Department of Computer Engineering (EGCO), Faculty of Engineering, Mahidol University**  
*Senior Capstone Thesis Project (Two-Member Engineering Team)*

---

## 1. Quick Navigation for Team Members

| Directory / File | Description |
| :--- | :--- |
| **[`Our work/`](./Our%20work/)** | **Master Thesis Dossier**: Contains all 5 finalized LaTeX chapters (`chapter1.tex`–`chapter5.tex`), `references.bib` (25 verified citations), [`senior_thesis_complete_compendium_report.pdf`](./Our%20work/senior_thesis_complete_compendium_report.pdf), [`papers_launchpad.html`](./Our%20work/papers_launchpad.html), and supplementary two-member technical specs. |
| **[`00 EGCO Bachelor Thesis Template 2023 for Upload/`](./00%20EGCO%20Bachelor%20Thesis%20Template%202023%20for%20Upload/)** | **Overleaf / XeLaTeX Ready Package**: Official Mahidol University EGCO Bachelor Thesis Template pre-loaded with our finalized `main.tex`, `abstractENG.tex`, `abstractTH.tex`, `chapter1.tex`–`chapter5.tex`, and `references.bib`. |
| **[`src/`](./src/)** | **Core Engineering Implementation**:<br>• `src/dsp/`: Zero-phase 4th-order Butterworth filter ($0.5\text{--}40.0\text{ Hz}$), physiological energy gate ($0.005 \le \sigma_{raw} \le 0.50\text{ m/s}^2$), moving-RMS adaptive peak detector, and PRV feature extractor.<br>• `src/models/` & `src/benchmark/`: Supervised arrhythmia classification models and ANSI/AAMI EC57 multi-database evaluation suite.<br>• `src/contracts/`: Two-member JSON integration contract (`team_integration_guide.py`). |
| **[`docs/`](./docs/)** | Compiled PDF and HTML reports, including [`senior_thesis_complete_compendium_report.pdf`](./docs/senior_thesis_complete_compendium_report.pdf) and [`BENCHMARK_REPORT.md`](./docs/BENCHMARK_REPORT.md). |
| **[`public/`](./public/)** | Web dashboard, interactive MEMS SCG anatomy visualizer, and mobile PWA telemetry interface. |
| **[`tests/`](./tests/)** | Automated unit and benchmark verification suite (`pytest`). |

---

## 2. Two-Member Responsibility Split (100% Sternal SCG Scope)

1. **Member 1 — Mobile Sternal SCG Acquisition & Edge Digital Signal Processing (Tiers 1 & 2)**:
   - Smartphone 3-axis MEMS accelerometer telemetry ($100\text{--}200\text{ Hz}$) via iOS `CoreMotion` and Android `SensorManager` in resting supine postures.
   - Zero-phase 4th-order Butterworth bandpass filtering ($0.5\text{--}40.0\text{ Hz}$), absolute physiological energy gating ($0.005\text{ m/s}^2 \le \sigma_{raw} \le 0.50\text{ m/s}^2$), gravitational posture checks, and real-time stillness guidance.
   - Moving-RMS unipolar energy envelope extraction, adaptive sliding threshold peak detection ($\theta[n] = \mu_E[n] + 0.60\sigma_E[n]$), and time/frequency-domain PRV feature computation within a $<500\text{ ms}$ latency budget.

2. **Member 2 — Machine Learning Arrhythmia Classification & In-Silico Digital Twin Simulation (Tiers 3 & 4)**:
   - Clinical dataset curation (PhysioNet MIT-BIH Arrhythmia Database & WESAD Stress Dataset) and feature alignment.
   - Supervised machine learning arrhythmia & stress classification (Decision Tree, Random Forest, Logistic Regression, and XGBoost — achieving $89.4\%$ accuracy, $88.9\%$ sensitivity, $89.8\%$ specificity, and $0.887$ F1-score).
   - Coupled nonlinear Van der Pol cardiac oscillator modeling (`scipy.integrate.solve_ivp` RK45) driven by extracted user PRV metrics to visualize limit-cycle phase portraits and bifurcation stability under simulated autonomic stress.

---

## 3. Running Locally

```bash
# Run the web server & analytical dashboard
python3 app.py

# Run the automated verification test suite
pytest
```
