> 🧠 **Obsidian Hub**: [[_Brain/Projects/Final-Thesis|Final-Thesis]] · [[Antigravity/Final-Thesis/CONTEXT|Final-Thesis/CONTEXT.md]]

# Academic Research & Citation Flow Protocol for Future Thesis Chapters

This protocol defines the mandatory 5-step standard operating procedure for discovering, citing, and verifying peer-reviewed academic literature across all future chapters (specifically **Chapter 4: Results and Discussion** and **Chapter 5: Conclusion and Future Work**).

---

## 1. The 5-Step Academic Citation Pipeline

```
[Draft Chapter Text] ──▶ 1. Identify Unsubstantiated Claims / Models
                              │
                              ▼
                         2. Primary Source Discovery (via research / search_web)
                              │
                              ▼
                         3. BibTeX Normalization & Semantic Keying
                              │
                              ▼
                         4. In-Text `\cite{...}` Insertion
                              │
                              ▼
                         5. Bidirectional Zero-Orphan Verification Gate
```

---

### Step 1: Claim & Baseline Audit
Whenever writing or editing chapter text, immediately flag:
1. **Clinical & Epidemiological Claims**: Any mortality rate, disease prevalence, or diagnostic criteria (e.g., WHO statistics, AHA/ESC guidelines).
2. **Algorithms & Computational Models**: Any machine learning model (Random Forest, XGBoost), filter equation (Butterworth), or numerical solver (Runge-Kutta).
3. **Hardware & Sensing Modalities**: Any statement on sensor physical properties (sternal mechanical recoil, 3-axis MEMS accelerometry, dorso-ventral chest vibration).
4. **Benchmark Datasets**: Any public corpus (PhysioNet MIT-BIH, WESAD).
5. **Evaluation Metrics**: Any formal performance metric formula (F1-score, Cohen's Kappa, Sensitivity, Specificity, MAE, RMSE).

### Step 2: Primary Source Discovery Protocol
* **Primary Sources Only**: Always trace claims back to the original foundational paper (IEEE, ACM, Circulation, Nature, Elsevier). Never cite tertiary secondary summaries, commercial blogs, or class slides.
* **Tool Delegation**:
  * Use `search_web` with query syntax: `"<Title or core concept>" <Author> <Year> bibtex` or `"<Method>" seminal paper IEEE DOI`.
  * For deep multi-paper literature surveys, invoke the `research` subagent to compile findings into a structured markdown brief before drafting.

### Step 3: BibTeX Normalization & Semantic Keying
Add all discovered sources into `Our work/references.bib` following standard semantic keys:
* Format: `\cite{<firstAuthorLastName><publicationYear><keyword>}`
  * Examples: `breiman2001random`, `chen2016xgboost`, `taskforce1996heart`, `inan2015ballistocardiography`.
* Mandatory Fields:
  ```bibtex
  @article{author2026topic,
      author = "First M. Last and Second Author",
      title = "Accurate Full Title of the Paper",
      journal = "Journal or Conference Name",
      volume = "10",
      number = "2",
      pages = "100--115",
      year = "2026",
      DOI = "10.1109/XXXX.2026.XXXX"
  }
  ```

### Step 4: In-Text Citation Placement in LaTeX
* Insert `\cite{author2026topic}` directly following the noun phrase, algorithm name, or numerical finding (e.g., `XGBoost classifier \cite{chen2016xgboost}`).
* For comparative results in Chapter 4: Cite the benchmark baseline directly in the comparison tables (e.g., *Our Work vs. Tsamis et al. \cite{tsamis2017soft}*).

### Step 5: Bidirectional Zero-Orphan Verification Gate
Run a quick terminal check before finalizing any chapter milestone:
1. **Check for missing keys**: Ensure every `\cite{key}` in `chapter*.tex` actually exists in `references.bib`.
2. **Check for unused entries**: Ensure every entry in `references.bib` is cited in at least one chapter.
3. **Synchronize**: Copy updated `.tex` and `references.bib` files to `00 EGCO Bachelor Thesis Template 2023 for Upload/`.

---

## 2. Chapter 4 & 5 Completed Citation Verification Checklist

In **Chapter 4 (Results & Discussion)**:
- [x] Cited reference classification baselines when reporting our model's Sensitivity ($88.9\%$) and Specificity ($89.8\%$).
- [x] Cited clinical ECG error bounds when comparing our heart rate MAE ($2.14\text{ bpm} < 3.0\text{ bpm}$) against simultaneous reference ECG.
- [x] Cited Bland-Altman statistical agreement methodology (`bland1986statistical`).

In **Chapter 5 (Conclusion & Future Work)**:
- [x] Cited wearable edge-AI hardware for future on-device NPU acceleration (`warden2019tinyml`).
- [x] Sternal Seismocardiography (SCG) focus: Removed obsolete optical PPG/rPPG citations, achieving 100% 1-to-1 citation parity across all 25 peer-reviewed papers.
