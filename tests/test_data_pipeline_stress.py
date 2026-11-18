"""
Empirical Stress Test Suite for ECG Data Pipeline (Milestone 1).

Adversarial stress testing against src/data/ modules:
1. Signal Dimensions & Boundary Conditions:
   - Empty 1D and 2D signals
   - 1-sample signals
   - Mismatched / ragged channel lengths
   - Positive and negative out-of-bounds lead indices
2. Extreme Heart Rates & Frequencies:
   - Tachycardia (300 BPM)
   - Severe Bradycardia (20 BPM)
   - Filter Nyquist boundary conditions (fs <= 3.0 Hz)
3. Noise, NaN/Inf, and Drift Artifacts:
   - NaN / Inf contamination & propagation
   - Massive baseline wander
   - Constant / flatline zero-variance signals
4. Annotation Anomalies & Ordering:
   - Out-of-order annotations and negative RR interval calculation
   - Duplicate annotations
   - Non-beat auxiliary symbols and unmapped characters
   - Boundary beats and out-of-bounds padding errors
5. Interface Contract 1 Compliance Under Stress:
   - Strict output shapes, types, and dtypes
"""

import numpy as np
import pytest
import scipy.signal
import wfdb

from src.data.preprocessor import ECGPreprocessor, resample_indices
from src.data.segmenter import (
    AAMI_BEAT_MAP,
    AAMI_CLASS_TO_INT,
    NON_BEAT_SYMBOLS,
    compute_rhythm_features,
    extract_heartbeats,
    is_qrs_beat,
    map_symbol_to_aami,
    normalize_annotations,
)
from src.data.wfdb_streamer import WFDBStreamer


# =============================================================================
# 1. SIGNAL DIMENSIONS AND BOUNDARY HARNESS
# =============================================================================

class TestSignalDimensionsAndBoundaries:
    """Stress tests for empty, minimal, and malformed signal inputs."""

    def test_empty_1d_signal_preprocessor(self):
        """Verify preprocessor handles empty 1D array safely returning empty array."""
        pre = ECGPreprocessor()
        empty_sig = np.array([], dtype=np.float64)
        out = pre.preprocess(empty_sig, fs=360.0)
        assert out.shape == (0,)

    def test_empty_signal_extract_heartbeats(self):
        """Verify extract_heartbeats receives empty 1D signal and returns Contract 1 empty structure."""
        empty_sig = np.array([], dtype=np.float64)
        res = extract_heartbeats("rec_empty", empty_sig, fs=360.0, annotations=[])
        assert res["beats"].shape == (0, 1, 360)
        assert res["rhythm_features"].shape == (0, 4)
        assert res["labels"].shape == (0,)
        assert res["r_peaks"].shape == (0,)

    def test_empty_2d_signal_extract_heartbeats(self):
        """Verify extract_heartbeats receives empty 2D signal (0, 2) and returns Contract 1 empty structure."""
        empty_sig_2d = np.zeros((0, 2), dtype=np.float64)
        res = extract_heartbeats("rec_empty_2d", empty_sig_2d, fs=360.0, annotations=[])
        assert res["beats"].shape == (0, 1, 360)
        assert res["rhythm_features"].shape == (0, 4)
        assert res["labels"].shape == (0,)
        assert res["r_peaks"].shape == (0,)

    def test_single_sample_signal(self):
        """Test preprocessor and segmentation on a 1-sample signal."""
        pre = ECGPreprocessor()
        sig_1 = np.array([1.5], dtype=np.float64)
        out = pre.preprocess(sig_1, fs=360.0)
        assert out.shape == (1,)
        assert out[0] == 0.0  # detrended and normalized

        # extract_heartbeats without valid window
        res = extract_heartbeats("rec_1", sig_1, fs=360.0, annotations=[(0, "N")], allow_padding=False)
        assert res["beats"].shape == (0, 1, 360)

    def test_ragged_multi_lead_signal_handling(self):
        """Test handling of multi-lead signals with mismatched channel lengths."""
        sig_ragged = [np.random.randn(2000), np.random.randn(1000)]
        # numpy cannot convert inhomogeneous sequences to a 2D float array
        with pytest.raises(ValueError, match="inhomogeneous shape"):
            extract_heartbeats("rec_ragged", sig_ragged, fs=360.0, annotations=[(500, "N")])

    def test_lead_index_boundary_conditions(self):
        """Test lead selection when lead index is out of bounds."""
        sig_2lead = np.random.randn(1000, 2)
        # Positive out-of-bounds: quietly clamps to channel 1
        res_clamp = extract_heartbeats("rec_lead", sig_2lead, fs=360.0, annotations=[(500, "N")], lead=10)
        assert res_clamp["beats"].shape == (1, 1, 360)

        # Negative out-of-bounds: raises IndexError
        with pytest.raises(IndexError):
            extract_heartbeats("rec_lead", sig_2lead, fs=360.0, annotations=[(500, "N")], lead=-5)


# =============================================================================
# 2. EXTREME HEART RATES AND FREQUENCIES
# =============================================================================

class TestExtremeHeartRatesAndFrequencies:
    """Stress tests for extreme physiological rates and filter boundary frequencies."""

    def test_extreme_high_heart_rate_300_bpm(self):
        """Test pipeline behavior under extreme tachycardia / ventricular flutter (300 BPM)."""
        fs = 360.0
        rec_300, ann_300 = WFDBStreamer.create_synthetic_record(
            "hr300", duration_sec=10.0, fs=fs, hr_bpm=300.0, add_arrhythmias=True
        )
        assert len(ann_300.sample) > 40
        res = extract_heartbeats("hr300", rec_300.p_signal, fs=fs, annotations=ann_300)

        assert res["beats"].shape[0] > 40
        assert res["beats"].shape[1:] == (1, 360)
        assert not np.isnan(res["beats"]).any()
        assert not np.isnan(res["rhythm_features"]).any()

        # At 300 BPM, RR intervals must be around 0.20s (60 / 300)
        np.testing.assert_allclose(res["rhythm_features"][:, 0], 0.20, atol=0.08)

    def test_extreme_low_heart_rate_20_bpm(self):
        """Test pipeline behavior under severe bradycardia (20 BPM)."""
        fs = 360.0
        rec_20, ann_20 = WFDBStreamer.create_synthetic_record(
            "hr20", duration_sec=30.0, fs=fs, hr_bpm=20.0, add_arrhythmias=True
        )
        assert len(ann_20.sample) >= 9
        res = extract_heartbeats("hr20", rec_20.p_signal, fs=fs, annotations=ann_20)

        assert res["beats"].shape[0] >= 9
        assert res["beats"].shape[1:] == (1, 360)
        assert not np.isnan(res["beats"]).any()
        assert not np.isnan(res["rhythm_features"]).any()

        # At 20 BPM, RR intervals must be around 3.00s (60 / 20)
        np.testing.assert_allclose(res["rhythm_features"][:, 0], 3.00, atol=0.20)

    def test_bandpass_filter_low_nyquist_cutoff_bug(self):
        """
        Verify Nyquist clamping: When fs <= 3.0 Hz, Nyquist <= 1.5 Hz.
        The robust clamping logic clamps highcut strictly below Nyquist and above lowcut,
        allowing scipy.signal.butter to succeed without raising ValueError.
        """
        pre = ECGPreprocessor()
        sig = np.random.randn(200)
        out = pre.bandpass_filter(sig, fs=2.0)
        assert out.shape == (200,)
        assert np.all(np.isfinite(out))


# =============================================================================
# 3. NOISE, NAN/INF, AND DRIFT ARTIFACTS
# =============================================================================

class TestNoiseNaNInfAndDrift:
    """Stress tests for numerical stability with NaNs, Infs, and baseline drift."""

    def test_nan_contamination_propagates_to_all_beats(self):
        """
        Document empirical observation: A single NaN sample in a multi-thousand
        sample recording propagates via zero-phase IIR filtfilt, contaminating
        100% of extracted beats into all-NaN arrays.
        """
        fs = 360.0
        sig = np.random.randn(5000)
        sig[10] = np.nan  # Single NaN early in recording

        ann = [(500, "N"), (1500, "N"), (2500, "N")]
        res = extract_heartbeats("rec_nan", sig, fs=fs, annotations=ann)

        assert res["beats"].shape[0] == 3
        # 100% of extracted beats are contaminated with NaNs
        assert np.isnan(res["beats"]).all()

    def test_inf_contamination_produces_nans(self):
        """Test signal with Inf value."""
        fs = 360.0
        sig = np.random.randn(2000)
        sig[100] = np.inf

        ann = [(500, "N"), (1000, "N")]
        res = extract_heartbeats("rec_inf", sig, fs=fs, annotations=ann)
        assert np.isnan(res["beats"]).all()

    def test_massive_baseline_drift(self):
        """Test signal with extreme baseline drift (10,000 mV linear ramp)."""
        fs = 360.0
        t = np.arange(2000) / fs
        drift = 10000.0 * t + 500.0 * np.sin(2 * np.pi * 0.1 * t)
        pure = 1.2 * np.sin(2 * np.pi * 1.2 * t)
        raw = pure + drift

        ann = [(600, "N"), (1200, "N")]
        res = extract_heartbeats("rec_drift", raw, fs=fs, annotations=ann)

        assert res["beats"].shape[0] == 2
        assert not np.isnan(res["beats"]).any()
        # Per-beat z-score normalization must yield zero-mean, unit-variance beats
        for b in res["beats"]:
            assert abs(np.mean(b)) < 1e-4
            assert abs(np.std(b) - 1.0) < 1e-3

    def test_flatline_constant_signal(self):
        """Test flatline signal (all 0.0 mV). eps guard prevents ZeroDivisionError."""
        fs = 360.0
        sig_zero = np.zeros(2000, dtype=np.float64)
        ann = [(500, "N"), (1000, "N")]
        res = extract_heartbeats("rec_zero", sig_zero, fs=fs, annotations=ann)

        assert res["beats"].shape[0] == 2
        assert not np.isnan(res["beats"]).any()
        np.testing.assert_allclose(res["beats"], 0.0)


# =============================================================================
# 4. ANNOTATION ANOMALIES AND ORDERING
# =============================================================================

class TestAnnotationAnomaliesAndOrdering:
    """Stress tests for annotation parsing, ordering, and symbol mapping."""

    def test_out_of_order_annotations_vulnerability(self):
        """
        Verify that out-of-chronological-order annotations are sorted chronologically:
        1. compute_rhythm_features receives monotonic peaks, resulting in non-negative RR intervals.
        2. Beats, labels, rhythm_features, and r_peaks are returned in chronological order.
        """
        fs = 360.0
        sig = np.random.randn(2000)
        ann_out_of_order = [(800, "N"), (300, "N"), (1500, "N")]
        res = extract_heartbeats("rec_ooo", sig, fs=fs, annotations=ann_out_of_order)

        # Output is chronologically sorted
        np.testing.assert_array_equal(res["r_peaks"], [300, 800, 1500])
        rf = res["rhythm_features"]

        # Verify non-negativity
        assert not (rf[:, 0] < 0).any(), "Expected non-negative RR_pre"
        assert not (rf[:, 1] < 0).any(), "Expected non-negative RR_post"
        assert not (rf[:, 3] < 0).any(), "Expected non-negative RR_ratio"

    def test_duplicate_annotations_handling(self):
        """Test handling of duplicate annotations at identical sample indices."""
        peaks = np.array([500, 500, 800], dtype=int)
        feats = compute_rhythm_features(peaks, fs=360.0)
        assert feats.shape == (3, 4)
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        # RR interval between identical peaks is 0.0s
        assert feats[0, 0] == 0.0

    def test_out_of_bounds_annotation_padding_crash(self):
        """
        Verify that out-of-bounds annotations do not crash when allow_padding=True.
        """
        fs = 360.0
        sig = np.random.randn(1000)
        res = extract_heartbeats("rec_pad_crash", sig, fs=fs, annotations=[(2000, "N")], allow_padding=True)
        assert res["beats"].shape == (0, 1, 360)

    def test_unmapped_and_invalid_symbols_filtered(self):
        """Test that unknown characters, numbers, and None are rejected gracefully."""
        fs = 360.0
        sig = np.random.randn(2000)
        ann = [
            (300, "INVALID_MARKER"),
            (500, ""),
            (700, 99999),
            (900, None),
            (1100, "N"),
        ]
        res = extract_heartbeats("rec_invalid", sig, fs=fs, annotations=ann)
        assert res["beats"].shape[0] == 1
        assert res["r_peaks"][0] == 1100
        assert res["labels"][0] == AAMI_CLASS_TO_INT["N"]

    def test_all_16_non_beat_symbols_filtered(self):
        """Verify all 16 AAMI non-beat symbols are completely filtered."""
        for sym in NON_BEAT_SYMBOLS:
            assert not is_qrs_beat(sym)
            assert map_symbol_to_aami(sym) is None


# =============================================================================
# 5. CONTRACT COMPLIANCE UNDER STRESS
# =============================================================================

class TestContractComplianceUnderStress:
    """Validate strict compliance with PROJECT.md Contract 1 under stress conditions."""

    def test_contract_1_shapes_and_types_under_stress(self):
        """Verify Contract 1 returns exact dictionary keys and tensor formats."""
        fs = 360.0
        rec, ann = WFDBStreamer.create_synthetic_record("c1_stress", duration_sec=8.0, fs=fs, hr_bpm=120.0)
        res = extract_heartbeats("c1_stress", rec.p_signal, fs=fs, annotations=ann)

        # 1. 'beats': np.ndarray of shape (N_beats, 1, 360), dtype float32
        assert "beats" in res
        assert isinstance(res["beats"], np.ndarray)
        assert res["beats"].ndim == 3
        assert res["beats"].shape[1:] == (1, 360)
        assert res["beats"].dtype == np.float32

        # 2. 'rhythm_features': np.ndarray of shape (N_beats, 4), dtype float32
        assert "rhythm_features" in res
        assert isinstance(res["rhythm_features"], np.ndarray)
        assert res["rhythm_features"].shape == (res["beats"].shape[0], 4)
        assert res["rhythm_features"].dtype == np.float32

        # 3. 'labels': np.ndarray of shape (N_beats,), integer encoded 0..4
        assert "labels" in res
        assert isinstance(res["labels"], np.ndarray)
        assert res["labels"].shape == (res["beats"].shape[0],)
        assert np.issubdtype(res["labels"].dtype, np.integer)
        assert set(res["labels"]).issubset({0, 1, 2, 3, 4})

        # 4. 'r_peaks': np.ndarray of shape (N_beats,), integer sample indices
        assert "r_peaks" in res
        assert isinstance(res["r_peaks"], np.ndarray)
        assert res["r_peaks"].shape == (res["beats"].shape[0],)
        assert np.issubdtype(res["r_peaks"].dtype, np.integer)
