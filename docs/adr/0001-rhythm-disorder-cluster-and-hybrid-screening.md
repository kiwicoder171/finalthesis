> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

---
status: accepted
date: 2026-11-26
---

# ADR-0001: Rhythm-Disorder Cluster Scope, Two-Stage Hybrid Screening, and Smartwatch Benchmarking

We scope Chapters 1–3 to a single physiological domain—the **Rhythm-Disorder Cluster** under Cardiac Arrhythmia (**Ectopic Beats [PVC/PAC]**, **Irregular Rhythm / AFib**, and **Rate Disorders [Bradycardia/Tachycardia]**)—using **Two-Stage Hybrid Screening** (deterministic HR rules for rate + explainable edge ML [`LR, RF, GBDT/XGBoost`] benchmarked against `Hybrid CNN-BiLSTM-Attention`), evaluated on patient-stratified clinical benchmarks (`MIT-BIH DS1/DS2`, `INCART`, `PTB-XL`, `CPSC 2018`, `Chapman-Shaoxing`) plus empirical smartphone SCG, and positioned quantitatively against published smartwatch PPG benchmarks with an optional complementary smartwatch pathway.

## Considered Options
- **Strictly 1 binary class (`Normal` vs. `Abnormal`)**: Safest literal reading of Progress Report No. 6, but under-utilizes the fact that resting HR and beat-to-beat PPI/PRV naturally separate rate disorders, ectopic beats, and irregular AFib-like rhythms within the exact same SCG feature space.
- **Adding structural/mechanical diseases (e.g., Heart Failure via PEP/LVET)**: Requires clinical SCG patient cohorts and waveform-morphology ground truth that exceed undergraduate recruitment scope.
- **Flat 4-class ML model**: Obscures deterministic clinical thresholds for Bradycardia ($<60\text{ bpm}$) and Tachycardia ($>100\text{ bpm}$) behind probabilistic model weights.

## Consequences
- Every Objective in Chapter 1 must carry explicit numeric targets across **Accuracy**, **Efficiency**, and **User-Friendliness** (`SUS ≥ 75`, setup `< 60 s`, first-attempt capture `≥ 90%`).
- Chapters 2 and 3 must cite peer-reviewed public smartwatch PPG benchmarks (e.g., Apple Heart Study, wearable PPG validation studies) in `Our work/references.bib`.
- Chapter 3 methodology and `BENCHMARK_REPORT.md` stay fully reconciled on patient-stratified evaluation and model lineup.
