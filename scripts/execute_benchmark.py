#!/usr/bin/env python3
"""
End-to-End Execution Script for Multi-Database ECG Rhythm Classification Benchmark.
Runs Method A (GBDT), Method B (1D-ResNet), and Method C (CNN-BiLSTM-Attention)
across MIT-BIH DS1/DS2, INCART, PTB-XL, CPSC 2018, and Chapman-Shaoxing.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.report_generator import generate_benchmark_report
from src.benchmark.runner import ECGBenchmarkRunner


def main() -> int:
    print("=" * 76)
    print("STANDARDIZED MULTI-DATABASE ECG RHYTHM CLASSIFICATION BENCHMARK")
    print("Protocol: ANSI/AAMI EC57 & de Chazal Inter-Patient Split (DS1 -> DS2)")
    print("=" * 76)

    runner = ECGBenchmarkRunner(
        mitbih_sampto=43200,  # 2 minutes per record across 28 records (~1,600+ clinical beats)
        incart_sampto=30840,  # 2 minutes per record across 5 INCART 12-lead records
        random_state=42,
    )

    results = runner.run_benchmarks(deep_epochs=10)
    generate_benchmark_report(results, output_paths=["docs/BENCHMARK_REPORT.md", "BENCHMARK_REPORT.md"])

    print("\n[RANKINGS SUMMARY]")
    for r in results["rankings"]:
        print(
            f"  #{r['rank']} {r['short_name']:<30} | "
            f"MIT-BIH DS2 Macro F1: {r['mitbih_ds2_macro_f1']:.4f} | "
            f"Acc: {r['mitbih_ds2_accuracy']*100:.2f}% | "
            f"Cross-DB Macro F1: {r['cross_db_mean_macro_f1']:.4f} | "
            f"Score: {r['clinical_composite_score']:.4f} | "
            f"Latency: {r['inference_latency_ms_per_beat']:.3f} ms/beat"
        )

    winner = results["winner"]
    print("\n" + "=" * 76)
    print(f"WINNING METHOD: {winner['display_name']}")
    print(f"  - MIT-BIH DS2 Macro F1 : {winner['mitbih_ds2_macro_f1']:.4f}")
    print(f"  - Cross-DB Mean Macro F1: {winner['cross_db_mean_macro_f1']:.4f}")
    print(f"  - VEB Sensitivity (Se_V): {winner['mitbih_ds2_se_v']*100:.2f}%")
    print(f"  - SVEB Sensitivity (Se_S): {winner['mitbih_ds2_se_s']*100:.2f}%")
    print(f"  - Report Written To     : docs/BENCHMARK_REPORT.md & BENCHMARK_REPORT.md")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
