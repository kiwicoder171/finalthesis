"""
Unit and End-to-End Integration Tests for Milestone 5 & 6:
Standardized ECG Benchmarking Engine, ANSI/AAMI EC57 Metrics, and Winner Selection.
"""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.benchmark.metrics import AAMI_CLASSES, compute_confusion_matrix, evaluate_ec57_metrics
from src.benchmark.report_generator import generate_benchmark_report
from src.benchmark.runner import ECGBenchmarkRunner, detect_r_peaks_and_annotate


class TestEC57MetricsEngine:
    def test_confusion_matrix_and_class_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4], dtype=int)
        y_pred = np.array([0, 0, 1, 1, 0, 2, 2, 3, 4], dtype=int)
        res = evaluate_ec57_metrics(y_true, y_pred)

        cm = res["confusion_matrix"]
        assert cm.shape == (5, 5)
        assert cm[0, 0] == 2
        assert cm[2, 2] == 2  # VEB 100% TP
        assert res["class_metrics"]["V"]["Se"] == 1.0
        assert res["class_metrics"]["V"]["+P"] == 1.0
        assert res["class_metrics"]["V"]["F1"] == 1.0
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_zero_division_safety_on_missing_classes(self):
        # Only classes N (0) and V (2) present
        y_true = np.array([0, 0, 0, 2, 2], dtype=int)
        y_pred = np.array([0, 0, 0, 0, 0], dtype=int)  # Predicts 0 V beats
        res = evaluate_ec57_metrics(y_true, y_pred)

        assert res["class_metrics"]["V"]["Se"] == 0.0
        assert res["class_metrics"]["V"]["+P"] == 0.0
        assert res["class_metrics"]["V"]["F1"] == 0.0
        assert res["class_metrics"]["F"]["support"] == 0
        assert np.isfinite(res["macro_f1"])

    def test_gross_vs_record_average_aggregation(self):
        y_true = np.array([0, 0, 2, 2, 0, 0, 1, 1], dtype=int)
        y_pred = np.array([0, 0, 2, 0, 0, 0, 1, 1], dtype=int)
        rec_ids = np.array(["100", "100", "100", "100", "101", "101", "101", "101"])

        res = evaluate_ec57_metrics(y_true, y_pred, record_ids=rec_ids)
        assert res["average_metrics"] is not None
        avg_c = res["average_metrics"]["class_metrics"]
        # Record 101 has 0 V beats, so only Record 100 contributes to V average Se (0.5)
        assert avg_c["V"]["records_with_support"] == 1
        assert abs(avg_c["V"]["Se_avg"] - 0.5) < 1e-5
        # Record 100 has 0 S beats, so only Record 101 contributes to S average Se (1.0)
        assert avg_c["S"]["records_with_support"] == 1
        assert abs(avg_c["S"]["Se_avg"] - 1.0) < 1e-5


class TestBenchmarkRunnerAndArtifacts:
    def test_detect_r_peaks_and_annotate(self):
        t = np.linspace(0, 10.0, 1000)
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 8.0 * t)
        anns = detect_r_peaks_and_annotate(sig, fs=100.0, record_seed=1)
        assert len(anns) >= 3
        for idx, sym in anns:
            assert 0 <= idx < 1000
            assert sym in ("N", "A", "V", "F")

    def test_Verify_generated_benchmark_artifacts(self):
        json_path = Path("data/benchmarks/benchmark_results.json")
        report_path = Path("docs/BENCHMARK_REPORT.md")
        if json_path.exists() and report_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "winner" in data
            assert "rankings" in data
            assert len(data["rankings"]) == 3
            assert "Method_A_GBDT" in data["methods"]
            assert "Method_B_ResNet1D" in data["methods"]
            assert "Method_C_CNN_BiLSTM_Attn" in data["methods"]

            md_text = report_path.read_text(encoding="utf-8")
            assert "Overall Method Ranking" in md_text
            assert "ANSI/AAMI EC57" in md_text
