"""
Empirical Challenge Test Suite for Milestone 1: Data Pipeline.
Authored and executed by Challenger 2 (replacement).

Validates:
1. Streaming and local caching resilience under offline & simulated network failure.
2. PROJECT.md Contract 1 invariant shapes ((N, 1, 360)), types, and boundary conditions.
3. Rhythm features finiteness, non-negativity (for intervals and ratio), and scaling.
4. ANSI/AAMI EC57 label bounds {0, 1, 2, 3, 4} with zero leakage across 19 beat + 16 non-beat + corrupt symbols.
5. Concrete reproduction tests for identified edge cases and bugs.
"""

from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
import wfdb

from src.data.ecg_catalog import ECGCatalog
from src.data.preprocessor import ECGPreprocessor, resample_indices
from src.data.segmenter import (
    AAMI_BEAT_MAP,
    AAMI_CLASS_TO_INT,
    NON_BEAT_SYMBOLS,
    compute_rhythm_features,
    extract_heartbeats,
    is_qrs_beat,
    map_symbol_to_aami,
)
from src.data.wfdb_streamer import WFDBStreamer


# =============================================================================
# CHALLENGE SUITE 1: STREAMING & CACHING RESILIENCE
# =============================================================================

class TestStreamingAndCachingResilience:
    """Stress tests for offline fallback and local caching integrity."""

    def test_mitbih_offline_simulated_network_failure(self):
        """Simulate complete socket/HTTP failure during PhysioNet streaming."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir, timeout_sec=2)

            with patch("wfdb.rdrecord", side_effect=ConnectionResetError("Simulated network drop")):
                with patch("wfdb.rdann", side_effect=ConnectionResetError("Simulated network drop")):
                    # 1. With fallback_to_synthetic=True: Must succeed gracefully
                    rec, ann = streamer.stream_mitbih(
                        "100",
                        sampfrom=0,
                        sampto=3600,
                        use_cache=False,
                        fallback_to_synthetic=True,
                    )
                    assert isinstance(rec, wfdb.Record)
                    assert isinstance(ann, wfdb.Annotation)
                    assert rec.p_signal.shape == (3600, 2)
                    assert rec.fs == 360
                    assert len(ann.sample) > 0

                    # 2. With fallback_to_synthetic=False: Must raise ConnectionError
                    with pytest.raises(ConnectionError, match="Failed to stream MIT-BIH record"):
                        streamer.stream_mitbih(
                            "100",
                            sampfrom=0,
                            sampto=3600,
                            use_cache=False,
                            fallback_to_synthetic=False,
                        )

    def test_all_five_databases_offline_fallback(self):
        """Verify all 5 database streaming endpoints gracefully handle offline conditions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir, timeout_sec=1)

            with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
                with patch("wfdb.rdrecord", side_effect=OSError("Network unreachable")):
                    with patch("wfdb.rdann", side_effect=OSError("Network unreachable")):
                        # 1. MIT-BIH
                        rec_mit, ann_mit = streamer.stream_mitbih("101", fallback_to_synthetic=True)
                        assert rec_mit.p_signal.shape[1] == 2
                        assert rec_mit.fs == 360

                        # 2. PTB-XL (100 Hz and 500 Hz)
                        rec_ptb100 = streamer.stream_ptbxl("00001", sampling_rate=100, fallback_to_synthetic=True)
                        assert rec_ptb100.p_signal.shape[1] == 12
                        assert rec_ptb100.fs == 100

                        rec_ptb500 = streamer.stream_ptbxl("00002", sampling_rate=500, fallback_to_synthetic=True)
                        assert rec_ptb500.p_signal.shape[1] == 12
                        assert rec_ptb500.fs == 500

                        # 3. CPSC 2018
                        cpsc_data = streamer.stream_cpsc2018("A0001", fallback_to_synthetic=True)
                        assert cpsc_data["signal"].shape[0] == 12
                        assert cpsc_data["fs"] == 500.0
                        assert cpsc_data["source"] == "synthetic_fallback"

                        # 4. Chapman-Shaoxing
                        chap_data = streamer.stream_chapman("JS00001", fallback_to_synthetic=True)
                        assert chap_data["signal"].shape[0] == 12
                        assert chap_data["fs"] == 500.0
                        assert chap_data["source"] == "synthetic_fallback"

                        # 5. INCART
                        rec_inc, ann_inc = streamer.stream_incart("I01", fallback_to_synthetic=True)
                        assert rec_inc.p_signal.shape[1] == 12
                        assert rec_inc.fs == 257

    def test_cache_hit_prevents_network_calls(self):
        """Verify that when files exist in cache, ZERO network calls are performed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)

            rec, ann = streamer.create_synthetic_record("201", duration_sec=5.0, fs=360.0)
            db_dir = Path(tmp_dir) / "mitdb"
            db_dir.mkdir(parents=True, exist_ok=True)

            wfdb.wrsamp("201", fs=rec.fs, units=rec.units, sig_name=rec.sig_name,
                        p_signal=rec.p_signal, fmt=["16", "16"], write_dir=str(db_dir))
            wfdb.wrann("201", "atr", sample=ann.sample, symbol=ann.symbol, write_dir=str(db_dir))

            with patch("urllib.request.urlopen", side_effect=AssertionError("Network call forbidden on cache hit!")):
                rec_hit, ann_hit = streamer.stream_mitbih("201", use_cache=True)
                assert rec_hit.record_name == "201"
                assert rec_hit.p_signal.shape == rec.p_signal.shape
                assert len(ann_hit.sample) == len(ann.sample)

    def test_cache_slicing_integrity(self):
        """Verify that partial sample slicing from cache respects sampfrom and sampto."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)

            rec, ann = streamer.create_synthetic_record("100", duration_sec=10.0, fs=360.0)
            db_dir = Path(tmp_dir) / "mitdb"
            db_dir.mkdir(parents=True, exist_ok=True)
            wfdb.wrsamp("100", fs=rec.fs, units=rec.units, sig_name=rec.sig_name,
                        p_signal=rec.p_signal, fmt=["16", "16"], write_dir=str(db_dir))
            wfdb.wrann("100", "atr", sample=ann.sample, symbol=ann.symbol, write_dir=str(db_dir))

            sliced_rec, sliced_ann = streamer.stream_mitbih("100", sampfrom=360, sampto=1800, use_cache=True)
            assert sliced_rec.p_signal.shape == (1440, 2)
            assert len(sliced_ann.sample) <= len(ann.sample)


# =============================================================================
# CHALLENGE SUITE 2: CONTRACT 1 INVARIANT SHAPES & VALID RANGES
# =============================================================================

class TestContract1InvariantShapes:
    """Empirical verification of Contract 1 shape, dtype, and dimension invariants."""

    @pytest.mark.parametrize("sig_len", [50, 120, 240, 359, 360, 500, 1000, 10000])
    def test_signal_lengths_beats_shape_invariant(self, sig_len):
        """Verify beats array is strictly (N, 1, 360), dtype float32 for valid non-empty signals."""
        fs = 360.0
        signal = np.sin(np.linspace(0, 10 * np.pi, sig_len)).astype(np.float32)
        mid = sig_len // 2
        ann = [(mid, "N")]

        for allow_pad in [False, True]:
            res = extract_heartbeats("test", signal, fs, ann, allow_padding=allow_pad)
            beats = res["beats"]

            assert isinstance(beats, np.ndarray)
            assert beats.ndim == 3
            assert beats.shape[1] == 1
            assert beats.shape[2] == 360
            assert beats.dtype == np.float32

            n_beats = beats.shape[0]
            assert res["rhythm_features"].shape == (n_beats, 4)
            assert res["rhythm_features"].dtype == np.float32
            assert res["labels"].shape == (n_beats,)
            assert res["r_peaks"].shape == (n_beats,)

    def test_multi_lead_orientation_invariants(self):
        """Verify (N_samples, N_leads) and (N_leads, N_samples) both produce (N, 1, 360)."""
        fs = 360.0
        n_samples = 2000
        n_leads = 12
        raw_sig = np.random.randn(n_samples, n_leads).astype(np.float32)
        ann = [(500, "N"), (1000, "V"), (1500, "S")]

        res1 = extract_heartbeats("rec1", raw_sig, fs, ann, lead=3)
        assert res1["beats"].shape == (3, 1, 360)

        res2 = extract_heartbeats("rec2", raw_sig.T, fs, ann, lead=3)
        assert res2["beats"].shape == (3, 1, 360)

    def test_sampling_rate_resampling_invariants(self):
        """Verify various sampling frequencies (100, 250, 500, 1000 Hz) yield exact (N, 1, 360) windows."""
        for orig_fs in [100.0, 250.0, 257.0, 500.0, 1000.0]:
            sig_duration = 5.0
            n_samples = int(orig_fs * sig_duration)
            t = np.arange(n_samples) / orig_fs
            sig = (np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(n_samples)).astype(np.float32)

            ann_samples = [int(round(sec * orig_fs)) for sec in [1.0, 2.0, 3.0, 4.0]]
            ann = [(s, "N") for s in ann_samples]

            res = extract_heartbeats("resample_test", sig, orig_fs, ann, target_fs=360.0)
            beats = res["beats"]
            assert beats.shape[1:] == (1, 360)
            assert beats.dtype == np.float32

    def test_numerical_stability_flatline_and_extreme_amplitudes(self):
        """Stress test zero-variance flatline signal and extreme amplitudes for NaNs/Infs."""
        fs = 360.0
        ann = [(500, "N")]

        # Case A: Complete flatline
        flat_signal = np.full(1000, 5.0, dtype=np.float32)
        res_flat = extract_heartbeats("flat", flat_signal, fs, ann)
        assert np.all(np.isfinite(res_flat["beats"]))
        assert np.all(np.isfinite(res_flat["rhythm_features"]))

        # Case B: Huge dynamic range (1e6)
        huge_signal = (np.random.randn(1000) * 1e6).astype(np.float32)
        res_huge = extract_heartbeats("huge", huge_signal, fs, ann)
        assert np.all(np.isfinite(res_huge["beats"]))

        # Case C: Tiny dynamic range (1e-6)
        tiny_signal = (np.random.randn(1000) * 1e-6).astype(np.float32)
        res_tiny = extract_heartbeats("tiny", tiny_signal, fs, ann)
        assert np.all(np.isfinite(res_tiny["beats"]))


# =============================================================================
# CHALLENGE SUITE 3: RHYTHM FEATURES FINITENESS, NON-NEGATIVITY, SCALING
# =============================================================================

class TestRhythmFeaturesContract:
    """Empirical verification of rhythm feature boundaries, positivity, and scaling."""

    def test_rhythm_features_finite_and_types(self):
        """Verify rhythm features are strictly finite (np.isfinite) and float32."""
        fs = 360.0
        peaks = np.array([300, 620, 950, 1400, 1600, 2100, 2400], dtype=int)
        feats = compute_rhythm_features(peaks, fs=fs)

        assert feats.shape == (len(peaks), 4)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))

    def test_sorted_rhythm_intervals_and_ratio_non_negative(self):
        """Verify RR_pre (col 0), RR_post (col 1), and RR_ratio (col 3) are non-negative for sorted peaks."""
        fs = 360.0
        np.random.seed(42)
        r_intervals = np.random.uniform(0.3, 1.8, size=50)
        peaks = np.cumsum(np.round(r_intervals * fs)).astype(int)

        feats = compute_rhythm_features(peaks, fs=fs)

        rr_pre = feats[:, 0]
        rr_post = feats[:, 1]
        delta_rr = feats[:, 2]
        rr_ratio = feats[:, 3]

        assert np.all(rr_pre >= 0.0)
        assert np.all(rr_post >= 0.0)
        assert np.all(rr_ratio >= 0.0)
        np.testing.assert_allclose(delta_rr, rr_pre - rr_post, atol=1e-5)

    def test_rhythm_feature_scaling_is_in_seconds(self):
        """Verify that RR features are scaled in seconds, not sample counts or milliseconds."""
        fs = 360.0
        peaks = np.array([360, 720, 1080], dtype=int)
        feats = compute_rhythm_features(peaks, fs=fs)

        np.testing.assert_allclose(feats[:, 0], 1.0, atol=1e-4)
        np.testing.assert_allclose(feats[:, 1], 1.0, atol=1e-4)
        np.testing.assert_allclose(feats[:, 2], 0.0, atol=1e-4)
        np.testing.assert_allclose(feats[:, 3], 1.0, atol=1e-4)

    def test_rhythm_features_extreme_edge_cases(self):
        """Test N=0, N=1, and duplicate peaks."""
        fs = 360.0

        # N = 0
        f0 = compute_rhythm_features(np.array([], dtype=int), fs=fs)
        assert f0.shape == (0, 4)
        assert f0.dtype == np.float32

        # N = 1
        f1 = compute_rhythm_features(np.array([500], dtype=int), fs=fs)
        assert f1.shape == (1, 4)
        assert np.all(np.isfinite(f1))
        assert f1[0, 0] > 0 and f1[0, 1] > 0

        # Duplicate peaks
        f_dup = compute_rhythm_features(np.array([500, 500, 800], dtype=int), fs=fs)
        assert f_dup.shape == (3, 4)
        assert np.all(np.isfinite(f_dup))


# =============================================================================
# CHALLENGE SUITE 4: LABEL BOUNDS & ZERO LEAKAGE
# =============================================================================

class TestLabelBoundsAndZeroLeakage:
    """Stress tests verifying strict bounds {0, 1, 2, 3, 4} and zero leakage."""

    def test_aami_5_classes_strict_bounds(self):
        """Ensure all 19 AAMI mapped beat symbols map strictly into {0, 1, 2, 3, 4}."""
        mapped_values = set(AAMI_CLASS_TO_INT.values())
        assert mapped_values == {0, 1, 2, 3, 4}

        for sym, expected_class in AAMI_BEAT_MAP.items():
            assert is_qrs_beat(sym) is True
            super_cls = map_symbol_to_aami(sym)
            assert super_cls in ["N", "S", "V", "F", "Q"]
            int_cls = AAMI_CLASS_TO_INT[super_cls]
            assert int_cls in {0, 1, 2, 3, 4}

    def test_all_16_non_beat_symbols_produce_zero_labels(self):
        """Verify non-beat markers produce 0 beats and never leak into labels."""
        fs = 360.0
        signal = np.random.randn(5000).astype(np.float32)
        non_beat_ann = [(i * 200 + 300, sym) for i, sym in enumerate(NON_BEAT_SYMBOLS)]

        res = extract_heartbeats("non_beat_test", signal, fs, non_beat_ann)
        assert res["beats"].shape[0] == 0
        assert res["labels"].shape[0] == 0
        assert res["rhythm_features"].shape[0] == 0
        assert res["r_peaks"].shape[0] == 0

    def test_corrupted_and_unmapped_symbols_rejection(self):
        """Verify unmapped, corrupt, whitespace, or invalid symbols are rejected with zero leakage."""
        fs = 360.0
        signal = np.random.randn(5000).astype(np.float32)
        invalid_symbols = ["X", "Z", "UNKNOWN", "123", "", " ", None, "@#$", "\t", "\n"]
        ann = [(i * 300 + 400, sym) for i, sym in enumerate(invalid_symbols)]

        res = extract_heartbeats("corrupt_test", signal, fs, ann)
        assert res["beats"].shape[0] == 0
        assert res["labels"].shape[0] == 0

    def test_mixed_stream_label_integrity_and_alignment(self):
        """Verify mixed stream with valid beats, non-beats, and corrupt symbols preserves strict 1:1 alignment."""
        fs = 360.0
        signal = np.random.randn(8000).astype(np.float32)

        mixed_ann = [
            (500, "N"),
            (600, "+"),
            (860, "A"),
            (1100, "~"),
            (1220, "V"),
            (1300, "CORRUPT"),
            (1580, "F"),
            (1940, "/"),
            (2000, "|"),
            (2300, "L"),
            (2660, "E"),
        ]

        res = extract_heartbeats("mixed_test", signal, fs, mixed_ann)
        beats = res["beats"]
        labels = res["labels"]
        rf = res["rhythm_features"]
        peaks = res["r_peaks"]

        expected_labels = [0, 1, 2, 3, 4, 0, 2]

        assert beats.shape[0] == len(expected_labels)
        assert len(labels) == len(expected_labels)
        assert rf.shape[0] == len(expected_labels)
        assert len(peaks) == len(expected_labels)

        np.testing.assert_array_equal(labels, expected_labels)
        assert set(labels).issubset({0, 1, 2, 3, 4})


# =============================================================================
# CHALLENGE SUITE 5: EMPIRICAL DEFECT DEMONSTRATIONS (EXPECTED FAILURES)
# =============================================================================

class TestEmpiricalDefectsDemonstration:
    """Explicit demonstration of edge cases and their contract-compliant behavior post-remediation."""

    def test_defect_1_empty_signal_crashes(self):
        """Fixed Defect 1: extract_heartbeats returns Contract 1 empty structure (0, 1, 360) on empty signal."""
        fs = 360.0
        empty_sig = np.zeros(0, dtype=np.float32)
        res = extract_heartbeats("empty", empty_sig, fs, [])
        assert res["beats"].shape == (0, 1, 360)
        assert res["rhythm_features"].shape == (0, 4)
        assert res["labels"].shape == (0,)
        assert res["r_peaks"].shape == (0,)

    def test_defect_2_out_of_bounds_padding_crashes(self):
        """Fixed Defect 2: extract_heartbeats with allow_padding=True handles out-of-bounds peak safely."""
        fs = 360.0
        sig = np.random.randn(1000).astype(np.float32)
        ann = [(5000, "N")]
        res = extract_heartbeats("oob", sig, fs, ann, allow_padding=True)
        assert res["beats"].shape == (0, 1, 360)

    def test_defect_3_unsorted_annotations_yield_negative_intervals(self):
        """Fixed Defect 3: Unsorted annotations are sorted and yield non-negative RR intervals and ratios."""
        fs = 360.0
        sig = np.random.randn(5000).astype(np.float32)
        unsorted_ann = [(1500, "N"), (800, "N"), (2500, "N")]
        res = extract_heartbeats("unsorted", sig, fs, unsorted_ann)
        np.testing.assert_array_equal(res["r_peaks"], [800, 1500, 2500])
        rf = res["rhythm_features"]
        assert not (rf[:, 0] < 0).any(), "Expected non-negative RR_pre"
        assert not (rf[:, 1] < 0).any(), "Expected non-negative RR_post"
        assert not (rf[:, 3] < 0).any(), "Expected non-negative RR_ratio"

    def test_defect_4_synthetic_record_lacks_fmt_for_wrsamp_cache(self):
        """Fixed Defect 4: Synthetic record has fmt=['16', ...], allowing wfdb.wrsamp caching without crash."""
        s = WFDBStreamer()
        rec, ann = s.create_synthetic_record("test_synth", duration_sec=10.0, fs=360.0)
        assert rec.fmt == ["16", "16"]
        assert rec.adc_gain == [200.0, 200.0]
        assert rec.baseline == [0, 0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            wfdb.wrsamp(
                "test_synth",
                fs=rec.fs,
                units=rec.units,
                sig_name=rec.sig_name,
                p_signal=rec.p_signal,
                fmt=rec.fmt,
                adc_gain=rec.adc_gain,
                baseline=rec.baseline,
                write_dir=tmp_dir,
            )
            r_read = wfdb.rdrecord(f"{tmp_dir}/test_synth")
            assert r_read.p_signal.shape == rec.p_signal.shape
