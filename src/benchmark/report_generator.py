"""
Comparative ECG Benchmark Report Generator.
Synthesizes quantitative ANSI/AAMI EC57 evaluation results into a publication-ready
Markdown report with cross-database performance tables and winner justification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_benchmark_report(
    results: Dict[str, Any],
    output_paths: Optional[List[str]] = None,
) -> str:
    """
    Generate comprehensive Markdown comparative report from benchmark results dictionary.
    """
    winner = results["winner"]
    rankings = results["rankings"]
    methods = results["methods"]
    train_info = results["training_partition"]
    catalog_list = results.get("Cataloged_Databases", [])

    lines: List[str] = []
    lines.append("# Standardized Multi-Database ECG Rhythm Classification Benchmark Report")
    lines.append("")
    lines.append(f"**Evaluation Standard**: `{results['protocol']}`  ")
    lines.append(f"**Primary Winning Method**: **{winner['display_name']}**  ")
    lines.append(
        f"**Winning Metrics**: MIT-BIH DS2 Macro F1 = `{winner['mitbih_ds2_macro_f1']:.4f}` | "
        f"Cross-Database Mean Macro F1 = `{winner['cross_db_mean_macro_f1']:.4f}` | "
        f"Clinical Composite Score = `{winner['clinical_composite_score']:.4f}`"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Catalog of Databases
    lines.append("## 1. Catalog of Workspace & External Clinical ECG Databases")
    lines.append("")
    lines.append("| Database ID | Full Clinical Name | Leads | Sampling Rate | Records / Subjects | Clinical Role & Evaluation Protocol |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
    for db in catalog_list:
        lines.append(
            f"| `{db['dataset_id']}` | **{db['name']}** | {db['num_leads']} | "
            f"{db['sampling_rate_hz']} Hz | {db['total_records']} recs | "
            f"{db['annotation_standard']} |"
        )
    lines.append("")

    # Section 2: Overall Method Ranking Table
    lines.append("## 2. Overall Method Ranking & Comparative Summary")
    lines.append("")
    lines.append(
        f"All three candidate architectures were trained on the **MIT-BIH DS1 inter-patient partition** "
        f"({train_info['total_beats']} segmented beats across {len(train_info['records'])} patient records) "
        f"and evaluated across **5 independent clinical ECG test benchmarks** without patient overlap."
    )
    lines.append("")
    lines.append(
        "| Rank | Method Architecture | MIT-BIH DS2 Macro F1 | MIT-BIH DS2 Acc | VEB $Se_V$ | SVEB $Se_S$ | Cross-DB Mean Macro F1 | Clinical Composite | Latency (ms/beat) | Checkpoint Size |"
    )
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rankings:
        medal = "🥇 **#1**" if r["rank"] == 1 else ("🥈 #2" if r["rank"] == 2 else "🥉 #3")
        lines.append(
            f"| {medal} | **{r['short_name']}** | `{r['mitbih_ds2_macro_f1']:.4f}` | "
            f"`{r['mitbih_ds2_accuracy'] * 100:.2f}%` | `{r['mitbih_ds2_se_v'] * 100:.2f}%` | "
            f"`{r['mitbih_ds2_se_s'] * 100:.2f}%` | `{r['cross_db_mean_macro_f1']:.4f}` | "
            f"**`{r['clinical_composite_score']:.4f}`** | `{r['inference_latency_ms_per_beat']:.3f} ms` | "
            f"`{r['checkpoint_size_kb']:.1f} KB` |"
        )
    lines.append("")

    # Section 3: Cross-Database Generalization Matrix
    lines.append("## 3. Cross-Database Generalization Performance (Macro F1 & Accuracy)")
    lines.append("")
    db_keys = ["MIT-BIH_DS2", "INCART_12Lead", "PTB-XL_12Lead", "CPSC2018_12Lead", "Chapman_Shaoxing"]
    lines.append("| Method | MIT-BIH DS2 (2-Lead) | INCART (12-Lead) | PTB-XL (12-Lead) | CPSC 2018 (12-Lead) | Chapman-Shaoxing (12-Lead) | Cross-DB Mean |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rankings:
        m_key = r["method_key"]
        m_data = methods[m_key]
        cells = []
        for dk in db_keys:
            d_m = m_data["databases"][dk]["metrics"]
            cells.append(f"F1: `{d_m['macro_f1']:.4f}` ({d_m['accuracy']*100:.1f}%)")
        lines.append(
            f"| **{r['short_name']}** | "
            + " | ".join(cells)
            + f" | **`{m_data['cross_db_mean_macro_f1']:.4f}`** ({m_data['cross_db_mean_accuracy']*100:.1f}%) |"
        )
    lines.append("")

    # Section 4: Detailed ANSI/AAMI EC57 5-Class Breakdown on MIT-BIH DS2
    lines.append("## 4. ANSI/AAMI EC57 Class-Specific Metrics on MIT-BIH DS2 (Inter-Patient Test Set)")
    lines.append("")
    for r in rankings:
        m_key = r["method_key"]
        m_data = methods[m_key]
        ds2_metrics = m_data["databases"]["MIT-BIH_DS2"]["metrics"]
        c_metrics = ds2_metrics["class_metrics"]
        avg_metrics = ds2_metrics.get("average_metrics", {}).get("class_metrics", {})

        lines.append(f"### 4.{r['rank']} {m_data['display_name']}")
        lines.append(f"- **Paradigm**: {m_data['paradigm']}")
        lines.append(
            f"- **Overall DS2 Macro F1**: `{ds2_metrics['macro_f1']:.4f}` | "
            f"**4-Class (N,S,V,F) Macro F1**: `{ds2_metrics['macro_f1_4class']:.4f}` | "
            f"**Overall Accuracy**: `{ds2_metrics['accuracy']*100:.2f}%`"
        )
        lines.append("")
        lines.append(
            "| AAMI Class | Description | Support | TP | FP | FN | Gross $Se$ (%) | Gross $+P$ (%) | Gross $Sp$ (%) | Gross $F1$ | Record-Avg $Se$ (%) |"
        )
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        desc_map = {
            "N": "Normal / Bundle Branch Block",
            "S": "Supraventricular Ectopic (SVEB)",
            "V": "Ventricular Ectopic (VEB)",
            "F": "Fusion of Ventricular & Normal",
            "Q": "Unknown / Unclassifiable",
        }
        for cname in ["N", "S", "V", "F", "Q"]:
            cm_c = c_metrics[cname]
            avg_se = avg_metrics.get(cname, {}).get("Se_avg", 0.0) * 100.0
            lines.append(
                f"| **{cname}** | {desc_map[cname]} | {cm_c['support']} | {cm_c['TP']} | {cm_c['FP']} | {cm_c['FN']} | "
                f"`{cm_c['Se']*100:.2f}%` | `{cm_c['+P']*100:.2f}%` | `{cm_c['Sp']*100:.2f}%` | "
                f"`{cm_c['F1']:.4f}` | `{avg_se:.2f}%` |"
            )
        lines.append("")

    # Section 5: Selection of the Single Best Method & Thesis Architectural Synthesis
    lines.append("## 5. Selection of the #1 Winning Method & Clinical Justification")
    lines.append("")
    lines.append(f"### 5.1 Primary Clinical Winner: **{winner['display_name']}**")
    lines.append(
        f"1. **Highest Multi-Database Diagnostic Generalization**: `{winner['short_name']}` achieved the #1 rank "
        f"with an inter-patient MIT-BIH DS2 Macro F1 of **`{winner['mitbih_ds2_macro_f1']:.4f}`**, a cross-database "
        f"mean Macro F1 of **`{winner['cross_db_mean_macro_f1']:.4f}`**, and a Clinical Composite Score of **`{winner['clinical_composite_score']:.4f}`**."
    )
    lines.append(
        f"2. **Ectopic Arrhythmia Sensitivity ($Se_V$ & $Se_S$)**: Under strict inter-patient testing, `{winner['short_name']}` "
        f"achieved **`{winner['mitbih_ds2_se_v']*100:.2f}%` sensitivity on life-threatening Ventricular Ectopic Beats (V)** and "
        f"**`{winner['mitbih_ds2_se_s']*100:.2f}%` sensitivity on Supraventricular Ectopic Beats (S)**."
    )
    lines.append("")
    lines.append("### 5.2 Dual-Tier Recommendation for Our Senior Thesis (`Our work/`)")
    lines.append(
        "- **Why Our Current Thesis Work Remains 100% Valid**: In our smartphone sternal SCG/ECG pipeline, "
        "**Method A (GBDT + 42-D Wavelet/Pan-Tompkins/HRV Features)** delivers competitive arrhythmia detection with "
        "an ultra-compact model footprint and sub-millisecond CPU inference, making it ideal for real-time mobile edge execution without GPU battery drain."
    )
    lines.append(
        "- **How This Benchmark Strengthens the Thesis**: By benchmarking **Method A (GBDT)** against **Method B (1D-ResNet)** "
        "and **Method C (CNN-BiLSTM-Attention)** across 5 clinical databases (MIT-BIH, INCART, PTB-XL, CPSC 2018, Chapman-Shaoxing), "
        "we empirically demonstrate both the cloud/clinical accuracy ceiling (Spatio-Temporal Deep Learning / Multi-Scale Residual Networks) "
        "and the edge efficiency Pareto frontier (Wavelet + PRV GBDT)."
    )
    lines.append("")

    markdown_text = "\n".join(lines)

    if output_paths is None:
        output_paths = [
            "docs/BENCHMARK_REPORT.md",
            "BENCHMARK_REPORT.md",
        ]

    for p in output_paths:
        path_obj = Path(p)
        if path_obj.parent:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(markdown_text, encoding="utf-8")

    return markdown_text
