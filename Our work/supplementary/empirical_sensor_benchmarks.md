> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Supplementary Technical Spec: Empirical Sensor Benchmarks & Datasets
> **Correlating Template Section**: Precursor empirical evidence for *Chapter 3 (Data & Performance Evaluation)* and *Chapter 4 (Experimental Results)*.

## 1. Empirical Hardware Test (Real iPhone Sternal SCG Recording)
- **Source**: [`data/raw_phyphox_test/Raw Data.csv`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/data/raw_phyphox_test/Raw Data.csv)
- **Sensor**: Apple iPhone internal 3-axis MEMS accelerometer placed flat on the sternum in a resting supine posture.
- **Recording Metrics**:
  - Total Duration: $37.54\text{ seconds}$
  - Total Samples: $3,737\text{ samples}$
  - Effective Sampling Rate: $99.52\text{ Hz}$
  - Baseline Gravity Offset: $9.342\text{ m/s}^2$
  - Dorso-Ventral Standard Deviation: $\sigma_{raw} = 0.0669\text{ m/s}^2$ (within valid physiological energy bounds $0.005 \le \sigma_{raw} \le 0.50\text{ m/s}^2$)
  - Detected Beats: $51\text{ beats}$
  - Mean Heart Rate: $90.1\text{ bpm}$
  - Signal Quality Status: `OPTIMAL` (usable for Member 2's classification & Van der Pol simulation)

## 2. Benchmark Corpus
- **Source**: [`data/training_features_corpus.csv`](file:///Users/kiwi/Desktop/Developer/Antigravity/Final-Thesis/data/training_features_corpus.csv)
- Generated 60-subject benchmark dataset pairing simulated cardiac stress and resting states alongside PhysioNet MIT-BIH and WESAD features to train and validate Member 2's supervised machine learning classifiers.
