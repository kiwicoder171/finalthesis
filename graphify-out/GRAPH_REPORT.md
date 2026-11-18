> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Graph Report - Final-Thesis  (2026-11-11)

## Corpus Check
- 299 files · ~246,017 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 69 file(s) not represented in the graph (top: .tex 21, .mat 11, .ttf 8)

## Summary
- 1109 nodes · 2156 edges · 68 communities (22 shown, 46 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 159 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- tests (81 nodes)
- models (49 nodes)
- tests (47 nodes)
- tests (37 nodes)
- scripts (34 nodes)
- data (33 nodes)
- dsp (29 nodes)
- benchmark (28 nodes)
- data (25 nodes)
- tests (25 nodes)
- tests (24 nodes)
- benchmark (23 nodes)
- models (23 nodes)
- tests (23 nodes)
- app (22 nodes)
- scripts (22 nodes)
- data (21 nodes)
- tests (18 nodes)
- tests (18 nodes)
- tests (17 nodes)
- scripts (17 nodes)
- scripts (17 nodes)
- scripts (17 nodes)
- explorer_m3 (16 nodes)
- explorer_m3 (16 nodes)
- scripts (16 nodes)
- data (16 nodes)
- contracts (16 nodes)
- explorer_m1_fix_1 (15 nodes)
- tests (15 nodes)
- tests (15 nodes)
- tests (14 nodes)
- tests (14 nodes)
- tests (14 nodes)
- tests (14 nodes)
- tests (14 nodes)
- tests (13 nodes)
- tests (13 nodes)
- dsp (12 nodes)
- tests (12 nodes)
- tests (11 nodes)
- dsp (11 nodes)
- tests (11 nodes)
- tests (10 nodes)
- tests (10 nodes)
- tests (10 nodes)
- tests (10 nodes)
- teamwork_explorer_survey_1 (9 nodes)
- static (9 nodes)
- static (9 nodes)
- tests (8 nodes)
- tests (8 nodes)
- tests (8 nodes)
- api (6 nodes)
- tests (6 nodes)
- explorer_m1_fix_1 (5 nodes)
- tests (4 nodes)
- scripts (3 nodes)
- vercel (3 nodes)
- static (2 nodes)
- static (2 nodes)

## God Nodes (most connected - your core abstractions)
1. `MethodBResNetClassifier` - 79 edges
2. `extract_heartbeats()` - 56 edges
3. `MethodAGBDTClassifier` - 44 edges
4. `WFDBStreamer` - 39 edges
5. `ECGPreprocessor` - 32 edges
6. `BaseECGClassifier` - 32 edges
7. `extract_features()` - 32 edges
8. `process_cardiac_recording()` - 24 edges
9. `MethodCHybridAttentionClassifier` - 24 edges
10. `MultiClassFocalLoss` - 22 edges

## Surprising Connections (you probably didn't know these)
- `TestECGCatalog` --uses--> `DatasetMetadata`  [INFERRED]
  tests/test_data_pipeline.py → src/data/ecg_catalog.py
- `TestPreprocessorFilterStabilityAdversarial` --uses--> `ECGPreprocessor`  [INFERRED]
  tests/test_challenger_m1_iter2.py → src/data/preprocessor.py
- `TestExtremeHeartRatesAndFrequencies` --uses--> `ECGPreprocessor`  [INFERRED]
  tests/test_data_pipeline_stress.py → src/data/preprocessor.py
- `TestSignalDimensionsAndBoundaries` --uses--> `ECGPreprocessor`  [INFERRED]
  tests/test_data_pipeline_stress.py → src/data/preprocessor.py
- `TestStreamerSerializationAndCachingAdversarial` --uses--> `WFDBStreamer`  [INFERRED]
  tests/test_challenger_m1_iter2.py → src/data/wfdb_streamer.py

## Import Cycles
- None detected.

## Communities (68 total, 46 thin omitted)

### Community 0 - "tests (81 nodes)"
Cohesion: 0.05
Nodes (5): MethodAGBDTClassifier, TestClassifierConvergenceAndTiming, TestContract2Robustness, TestModelTrainingAndInferenceStress, TestMethodAGBDTClassifier

### Community 1 - "models (49 nodes)"
Cohesion: 0.07
Nodes (6): BahdanauTemporalAttention, CNNBiLSTMAttention1D, MethodCHybridAttentionClassifier, _generate_synthetic_dataset(), TestMethodCArchitecture, TestMethodCClassifierContract

### Community 2 - "tests (47 nodes)"
Cohesion: 0.06
Nodes (4): BaseECGClassifier, extract_features(), TestFeatureExtractorAdversarialBoundaries, TestBaseECGClassifierInterface

### Community 3 - "tests (37 nodes)"
Cohesion: 0.07
Nodes (5): extract_heartbeats(), normalize_annotations(), TestSegmenterNumericalInvariantsAdversarial, TestNoiseNaNInfAndDrift, TestBeatSegmenter

### Community 4 - "scripts (34 nodes)"
Cohesion: 0.11
Nodes (29): at(), BASE, body, byDepth(), CHIP, D(), dialSvg, drawAxesReticle() (+21 more)

### Community 5 - "data (33 nodes)"
Cohesion: 0.14
Nodes (4): DatasetMetadata, resample_indices(), is_qrs_beat(), map_symbol_to_aami()

### Community 7 - "dsp (29 nodes)"
Cohesion: 0.14
Nodes (9): generate_benchmark_training_corpus(), filter_scg_respiration(), compute_peak_intervals(), detect_systolic_peaks(), process_cardiac_recording(), extract_time_domain_features(), compute_signal_quality(), run_tests() (+1 more)

### Community 8 - "benchmark (28 nodes)"
Cohesion: 0.14
Nodes (6): compute_confusion_matrix(), evaluate_ec57_metrics(), _metrics_from_cm(), detect_r_peaks_and_annotate(), TestBenchmarkRunnerAndArtifacts, TestEC57MetricsEngine

### Community 12 - "benchmark (23 nodes)"
Cohesion: 0.13
Nodes (3): main(), generate_benchmark_report(), ECGBenchmarkRunner

### Community 13 - "models (23 nodes)"
Cohesion: 0.11
Nodes (3): MultiClassFocalLoss, ResidualBlock1D, TestMultiClassFocalLoss

### Community 15 - "app (22 nodes)"
Cohesion: 0.13
Nodes (4): CardioServerHandler, get_lan_ip(), run_server(), generate_dashboard()

### Community 16 - "scripts (22 nodes)"
Cohesion: 0.17
Nodes (3): generate_thesis_compendium(), ThesisCompendiumPDF, wrap_text()

### Community 18 - "tests (18 nodes)"
Cohesion: 0.14
Nodes (3): ECGResNet1D, TestArchitectureAndTensorFlow, TestSingleClassAndSubsetEdgeCases

### Community 24 - "explorer_m3 (16 nodes)"
Cohesion: 0.14
Nodes (3): ECGResNet1D, MultiClassFocalLoss, ResidualBlock1D

### Community 26 - "scripts (16 nodes)"
Cohesion: 0.19
Nodes (8): apply_iir_filter(), butterworth_bandpass_coeffs(), compute_sampling_rate(), detect_systolic_peaks(), fft_power_spectrum(), fft(), load_data(), main()

### Community 28 - "contracts (16 nodes)"
Cohesion: 0.19
Nodes (6): member_c_quickstart_example(), member_d_quickstart_example(), generate_synthetic_scg(), test_contracts(), test_pipeline_real_and_synthetic(), test_quality_and_motion()

### Community 31 - "tests (15 nodes)"
Cohesion: 0.14
Nodes (3): generate_balanced_test_dataset(), generate_de_chazal_imbalanced_dataset(), TestImbalanceAndFocalLossRobustness

### Community 39 - "dsp (12 nodes)"
Cohesion: 0.24
Nodes (5): apply_zero_phase_filter(), design_butterworth_bandpass(), filter_ppg_cardiac(), filter_scg_cardiac(), test_filters()

### Community 43 - "dsp (11 nodes)"
Cohesion: 0.22
Nodes (5): compute_fft_power_spectrum(), _cooley_tukey_fft(), extract_frequency_domain_features(), extract_poincare_descriptors(), test_prv_metrics()

### Community 49 - "teamwork_explorer_survey_1 (9 nodes)"
Cohesion: 0.28
Nodes (3): verify_chapman_streaming(), verify_cpsc2018_streaming(), verify_ptbxl_streaming()

### Community 50 - "static (9 nodes)"
Cohesion: 0.22
Nodes (8): background_color, display, icons, name, orientation, short_name, start_url, theme_color

### Community 51 - "static (9 nodes)"
Cohesion: 0.22
Nodes (8): background_color, display, icons, name, orientation, short_name, start_url, theme_color

## Knowledge Gaps
- **1 isolated node(s):** `PYTHONPATH`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 491 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MethodBResNetClassifier` connect `tests (18 nodes)` to `tests (14 nodes)`, `tests (47 nodes)`, `tests (14 nodes)`, `tests (14 nodes)`, `tests (13 nodes)`, `tests (30 nodes)`, `benchmark (28 nodes)`, `models (12 nodes)`, `tests (24 nodes)`, `benchmark (23 nodes)`, `tests (10 nodes)`, `tests (10 nodes)`, `tests (18 nodes)`, `tests (8 nodes)`, `tests (5 nodes)`, `tests (15 nodes)`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MethodBResNetClassifier` (e.g. with `ECGBenchmarkRunner` and `TestBoundaryAndCornerCases`) actually correct?**
  _`MethodBResNetClassifier` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PYTHONPATH` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `tests (81 nodes)` be split into smaller, more focused modules?**
  _Cohesion score 0.05308641975308642 - nodes in this community are weakly interconnected._
- **Why does `extract_heartbeats()` connect `tests (37 nodes)` to `tests (14 nodes)`, `tests (14 nodes)`, `data (33 nodes)`, `benchmark (28 nodes)`, `data (25 nodes)`, `tests (12 nodes)`, `benchmark (23 nodes)`, `tests (11 nodes)`, `tests (10 nodes)`, `tests (10 nodes)`, `tests (8 nodes)`, `tests (4 nodes)`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `MethodAGBDTClassifier` (e.g. with `ECGBenchmarkRunner` and `TestClassifierConvergenceAndTiming`) actually correct?**
  _`MethodAGBDTClassifier` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `models (49 nodes)` be split into smaller, more focused modules?**
  _Cohesion score 0.07312925170068027 - nodes in this community are weakly interconnected._