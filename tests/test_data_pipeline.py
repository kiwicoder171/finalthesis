"""
Unit and Integration Test Suite for ECG Data Pipeline (Milestone 1).

Validates:
- Database Catalog and de Chazal Inter-Patient Split Partitions
- Signal Preprocessor (Butterworth Bandpass, Detrending, Resampling, Normalization)
- Beat Segmenter, AAMI EC57 5-Class Mapping, and PROJECT.md Contract 1
- WFDB Streamer, Caching, and Synthetic Fallback Engine
- Edge Cases (Empty beats, extreme frequencies, truncated signals, boundary beats)
"""

import math
from pathlib import Path
import tempfile
import numpy as np
import pytest
import scipy.signal
import wfdb

from src.data.ecg_catalog import (
    ALL_MITBIH_RECORDS,
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_RECORDS,
    DatasetMetadata,
    ECGCatalog,
)
from src.data.preprocessor import ECGPreprocessor, resample_indices
from src.data.segmenter import (
    AAMI_BEAT_MAP,
    AAMI_CLASS_TO_INT,
    INT_TO_AAMI_CLASS,
    NON_BEAT_SYMBOLS,
    compute_rhythm_features,
    extract_heartbeats,
    is_qrs_beat,
    map_symbol_to_aami,
    normalize_annotations,
)
from src.data.wfdb_streamer import WFDBStreamer


# =============================================================================
# 1. ECG CATALOG & DE CHAZAL INTER-PATIENT SPLIT TESTS
# =============================================================================

class TestECGCatalog:
    """Test suite for database catalog and partitioning."""

    def test_catalog_listing(self):
        datasets = ECGCatalog.list_datasets()
        assert "mitdb" in datasets
        assert "ptb-xl" in datasets
        assert "cpsc2018" in datasets
        assert "chapman" in datasets
        assert "incartdb" in datasets
        assert "wesad" in datasets
        assert "workspace_training_corpus" in datasets

    def test_dataset_metadata(self):
        meta = ECGCatalog.get_dataset("mitdb")
        assert isinstance(meta, DatasetMetadata)
        assert meta.sampling_rate_hz == 360.0
        assert meta.num_leads == 2
        assert meta.num_records == 48
        assert "N" in meta.classes_supported
        assert meta.is_external is True

        ptb = ECGCatalog.get_dataset("ptb-xl")
        assert ptb.num_leads == 12
        assert ptb.sampling_rate_hz == 100.0

    def test_de_chazal_partitions_validity(self):
        ds1, ds2, paced = ECGCatalog.get_de_chazal_split(strict_zero_overlap=False)

        # Exact record counts mandated by de Chazal et al. (2004)
        assert len(ds1) == 22, f"Expected 22 DS1 records, got {len(ds1)}"
        assert len(ds2) == 22, f"Expected 22 DS2 records, got {len(ds2)}"
        assert len(paced) == 4, f"Expected 4 paced records, got {len(paced)}"

        set_ds1 = set(ds1)
        set_ds2 = set(ds2)
        set_paced = set(paced)

        # Mutually disjoint partitions
        assert set_ds1.isdisjoint(set_ds2), "DS1 and DS2 have overlapping records!"
        assert set_ds1.isdisjoint(set_paced), "DS1 contains paced records!"
        assert set_ds2.isdisjoint(set_paced), "DS2 contains paced records!"

        # Total 48 records
        total_records = set_ds1 | set_ds2 | set_paced
        assert len(total_records) == 48, f"Expected 48 unique records, got {len(total_records)}"
        assert total_records == set(ALL_MITBIH_RECORDS)

    def test_de_chazal_strict_zero_overlap_mode(self):
        # In strict zero overlap mode, Record 202 is moved from DS2 to DS1
        ds1, ds2, paced = ECGCatalog.get_de_chazal_split(strict_zero_overlap=True)
        assert "201" in ds1
        assert "202" in ds1
        assert "202" not in ds2
        assert len(ds1) == 23
        assert len(ds2) == 21

    def test_patient_mapping(self):
        mapping = ECGCatalog.get_patient_mapping()
        assert mapping["100"] == "Patient_100"
        assert mapping["201"] == "Patient_201"
        assert mapping["202"] == "Patient_201", "Record 202 must map to Patient 201"

    def test_workspace_assets_scan(self):
        scan = ECGCatalog.scan_workspace_assets()
        assert "assets_found" in scan
        assert "training_features_corpus" in scan["assets_found"]
        assert "raw_phyphox_test" in scan["assets_found"]


# =============================================================================
# 2. SIGNAL PREPROCESSOR TESTS
# =============================================================================

class TestECGPreprocessor:
    """Test suite for ECG digital signal processing."""

    @pytest.fixture
    def sample_ecg_signal(self):
        fs = 360.0
        duration = 5.0  # 5 seconds
        t = np.arange(int(fs * duration)) / fs
        # Synthetic signal: 1.2 Hz fundamental + 0.1 Hz baseline drift + 60 Hz hum
        pure_ecg = 1.0 * np.sin(2 * np.pi * 1.2 * t)
        baseline_drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        mains_noise = 0.2 * np.sin(2 * np.pi * 60.0 * t)
        raw = pure_ecg + baseline_drift + mains_noise
        return raw, fs

    def test_bandpass_attenuates_low_and_high_frequencies(self, sample_ecg_signal):
        raw, fs = sample_ecg_signal
        pre = ECGPreprocessor(lowcut=0.5, highcut=45.0, filter_order=4)
        filtered = pre.bandpass_filter(raw, fs=fs)

        assert filtered.shape == raw.shape
        # Check standard deviation of filtered is lower due to removal of drift and hum
        assert np.std(filtered) < np.std(raw)

        # In frequency domain: power at 0.1 Hz and 60 Hz must be strongly attenuated
        fft_raw = np.abs(np.fft.rfft(raw))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(raw), 1 / fs)

        idx_01 = np.argmin(np.abs(freqs - 0.1))
        idx_60 = np.argmin(np.abs(freqs - 60.0))
        idx_12 = np.argmin(np.abs(freqs - 1.2))

        # Fundamental 1.2 Hz preserved
        assert fft_filt[idx_12] / fft_raw[idx_12] > 0.85
        # 0.1 Hz drift attenuated by at least 80%
        assert fft_filt[idx_01] / fft_raw[idx_01] < 0.20
        # 60 Hz mains attenuated by at least 80%
        assert fft_filt[idx_60] / fft_raw[idx_60] < 0.20

    def test_detrend(self):
        pre = ECGPreprocessor()
        t = np.linspace(0, 10, 1000)
        ramp = 5.0 * t + np.sin(t)
        detrended = pre.detrend(ramp)
        # Trend slope should be approximately zero
        slope, _ = np.polyfit(t, detrended, 1)
        assert abs(slope) < 1e-3

    def test_resampling_ratio(self):
        pre = ECGPreprocessor(target_fs=360.0)
        orig_fs = 100.0
        duration = 2.0
        sig_100hz = np.random.randn(int(orig_fs * duration))
        resampled = pre.resample(sig_100hz, orig_fs=orig_fs, target_fs=360.0)

        expected_len = int(round(len(sig_100hz) * (360.0 / 100.0)))
        assert len(resampled) == expected_len == 720

    def test_resample_indices(self):
        peaks_100hz = [0, 50, 100, 150]
        peaks_360hz = resample_indices(peaks_100hz, orig_fs=100.0, target_fs=360.0)
        expected = np.array([0, 180, 360, 540])
        np.testing.assert_array_equal(peaks_360hz, expected)

    def test_z_score_normalization(self):
        pre = ECGPreprocessor()
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        norm = pre.z_score_normalize(data)
        assert abs(np.mean(norm)) < 1e-6
        assert abs(np.std(norm) - 1.0) < 1e-4

    def test_full_pipeline_multi_channel(self):
        pre = ECGPreprocessor(target_fs=360.0, normalize=True)
        # 2 channels, 500 Hz, 3 seconds
        sig_2ch = np.random.randn(1500, 2)
        out = pre.preprocess(sig_2ch, fs=500.0)
        expected_len = int(round(1500 * (360.0 / 500.0)))
        assert out.shape == (expected_len, 2)
        assert out.dtype == np.float32


# =============================================================================
# 3. BEAT SEGMENTER & AAMI EC57 MAPPING TESTS
# =============================================================================

class TestBeatSegmenter:
    """Test suite for beat extraction and AAMI EC57 mapping."""

    def test_aami_19_symbols_mapping(self):
        # 1. Normal / Non-ectopic (N)
        for sym in ["N", "L", "R", "B", "e", "j", "n"]:
            assert map_symbol_to_aami(sym) == "N", f"Symbol '{sym}' failed to map to 'N'"
            assert is_qrs_beat(sym) is True

        # 2. Supraventricular Ectopic (S)
        for sym in ["A", "a", "J", "S"]:
            assert map_symbol_to_aami(sym) == "S", f"Symbol '{sym}' failed to map to 'S'"
            assert is_qrs_beat(sym) is True

        # 3. Ventricular Ectopic (V)
        for sym in ["V", "E", "r"]:
            assert map_symbol_to_aami(sym) == "V", f"Symbol '{sym}' failed to map to 'V'"
            assert is_qrs_beat(sym) is True

        # 4. Fusion (F)
        assert map_symbol_to_aami("F") == "F"
        assert is_qrs_beat("F") is True

        # 5. Unknown / Paced (Q)
        for sym in ["/", "f", "Q", "?"]:
            assert map_symbol_to_aami(sym) == "Q", f"Symbol '{sym}' failed to map to 'Q'"
            assert is_qrs_beat(sym) is True

    def test_non_beat_filtering(self):
        for sym in NON_BEAT_SYMBOLS:
            assert is_qrs_beat(sym) is False, f"Non-beat symbol '{sym}' was not filtered!"
            assert map_symbol_to_aami(sym) is None

    def test_compute_rhythm_features(self):
        # Peaks at 1.0s, 1.8s, 2.6s, 3.4s at fs=360
        fs = 360.0
        peaks = np.array([360, 648, 936, 1224], dtype=int)
        feats = compute_rhythm_features(peaks, fs=fs)

        assert feats.shape == (4, 4)
        assert feats.dtype == np.float32

        # In steady 0.8s rhythm:
        # RR_pre and RR_post should both be 0.8s
        # delta_RR should be 0.0
        # RR_ratio should be 1.0
        np.testing.assert_allclose(feats[:, 0], 0.8, atol=1e-3)
        np.testing.assert_allclose(feats[:, 1], 0.8, atol=1e-3)
        np.testing.assert_allclose(feats[:, 2], 0.0, atol=1e-3)
        np.testing.assert_allclose(feats[:, 3], 1.0, atol=1e-3)

    def test_contract_1_extract_heartbeats(self):
        # Create a realistic test recording using WFDB streamer synthetic generator
        rec, ann = WFDBStreamer.create_synthetic_record(
            record_name="test_rec",
            duration_sec=10.0,
            fs=360.0,
            num_leads=2,
            hr_bpm=75.0,
            add_arrhythmias=True,
        )

        res = extract_heartbeats(
            record_id="test_rec",
            raw_signal=rec.p_signal,
            fs=360.0,
            annotations=ann,
            target_fs=360.0,
            lead=0,
            pre_samples=120,
            post_samples=240,
        )

        # Check Contract 1 specifications:
        # 1. 'beats': np.ndarray of shape (N_beats, 1, 360), float32
        assert "beats" in res
        beats = res["beats"]
        assert isinstance(beats, np.ndarray)
        assert beats.ndim == 3
        assert beats.shape[1] == 1
        assert beats.shape[2] == 360
        assert beats.dtype == np.float32

        # 2. 'rhythm_features': np.ndarray of shape (N_beats, 4), float32
        assert "rhythm_features" in res
        rf = res["rhythm_features"]
        assert rf.shape == (beats.shape[0], 4)
        assert rf.dtype == np.float32

        # 3. 'labels': np.ndarray of shape (N_beats,), integer 0..4
        assert "labels" in res
        labels = res["labels"]
        assert labels.shape == (beats.shape[0],)
        assert np.issubdtype(labels.dtype, np.integer)
        assert all(0 <= lbl <= 4 for lbl in labels)

        # 4. 'r_peaks': np.ndarray of shape (N_beats,), sample indices
        assert "r_peaks" in res
        peaks = res["r_peaks"]
        assert peaks.shape == (beats.shape[0],)

        # Check beat normalization (per-beat mean ~ 0, std ~ 1)
        for i in range(len(beats)):
            b = beats[i, 0, :]
            assert abs(np.mean(b)) < 1e-4
            assert abs(np.std(b) - 1.0) < 1e-3

    def test_annotation_formats_compatibility(self):
        fs = 360.0
        signal = np.random.randn(3600)  # 10s

        # Format 1: List of (sample, symbol) tuples
        ann_tuples = [(500, "N"), (860, "V"), (1220, "A")]
        res1 = extract_heartbeats("f1", signal, fs, ann_tuples)
        assert res1["beats"].shape[0] == 3

        # Format 2: Dict format
        ann_dict = {"sample": [500, 860, 1220], "symbol": ["N", "V", "A"]}
        res2 = extract_heartbeats("f2", signal, fs, ann_dict)
        assert res2["beats"].shape[0] == 3

        # Format 3: wfdb.Annotation
        ann_obj = wfdb.Annotation("f3", "atr", sample=np.array([500, 860, 1220]), symbol=["N", "V", "A"])
        res3 = extract_heartbeats("f3", signal, fs, ann_obj)
        assert res3["beats"].shape[0] == 3

    def test_boundary_beats_omission(self):
        fs = 360.0
        signal = np.random.randn(1000)
        # R peak at sample 50 (too close to left boundary, 50 - 120 < 0)
        # R peak at sample 500 (valid, 500 - 120 = 380, 500 + 240 = 740 <= 1000)
        # R peak at sample 900 (too close to right boundary, 900 + 240 = 1140 > 1000)
        ann = [(50, "N"), (500, "N"), (900, "N")]

        # Default: allow_padding=False -> boundary beats omitted
        res_no_pad = extract_heartbeats("b1", signal, fs, ann, allow_padding=False)
        assert res_no_pad["beats"].shape[0] == 1
        assert res_no_pad["r_peaks"][0] == 500

        # With allow_padding=True -> all 3 beats kept with padding
        res_pad = extract_heartbeats("b2", signal, fs, ann, allow_padding=True)
        assert res_pad["beats"].shape[0] == 3


# =============================================================================
# 4. WFDB STREAMER & CACHING TESTS
# =============================================================================

class TestWFDBStreamer:
    """Test suite for WFDB streaming, caching, and synthetic fallback."""

    def test_synthetic_ecg_record_generation(self):
        rec, ann = WFDBStreamer.create_synthetic_record(
            record_name="synth_test",
            duration_sec=6.0,
            fs=360.0,
            num_leads=2,
            hr_bpm=72.0,
            add_arrhythmias=True,
        )

        assert isinstance(rec, wfdb.Record)
        assert isinstance(ann, wfdb.Annotation)
        assert rec.p_signal.shape == (int(6.0 * 360.0), 2)
        assert rec.fs == 360
        assert len(ann.sample) > 0
        assert len(ann.symbol) == len(ann.sample)

    def test_local_caching_mechanism(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)

            # Generate synthetic record and write to cache
            rec, ann = streamer.create_synthetic_record(record_name="100", duration_sec=5.0, fs=360.0)
            db_dir = Path(tmp_dir) / "mitdb"
            db_dir.mkdir(parents=True)

            wfdb.wrsamp(
                "100",
                fs=rec.fs,
                units=rec.units,
                sig_name=rec.sig_name,
                p_signal=rec.p_signal,
                fmt=rec.fmt,
                write_dir=str(db_dir),
            )
            wfdb.wrann(
                "100",
                "atr",
                sample=ann.sample,
                symbol=ann.symbol,
                write_dir=str(db_dir),
            )

            # Loading from cache
            rec_cached, ann_cached = streamer.stream_mitbih("100", use_cache=True)
            assert rec_cached.record_name == "100"
            assert rec_cached.p_signal.shape == rec.p_signal.shape
            assert len(ann_cached.sample) == len(ann.sample)

    def test_ptbxl_streamer_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            rec = streamer.stream_ptbxl("00001", sampling_rate=100, use_cache=True, fallback_to_synthetic=True)
            assert rec is not None
            assert rec.p_signal.ndim == 2
            assert rec.p_signal.shape[1] == 12  # 12 leads

    def test_cpsc2018_streamer_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            data = streamer.stream_cpsc2018("A0001", use_cache=True, fallback_to_synthetic=True)
            assert "signal" in data
            assert "fs" in data
            assert data["signal"].shape[0] == 12  # 12 leads

    def test_chapman_streamer_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            data = streamer.stream_chapman("JS00001", use_cache=True, fallback_to_synthetic=True)
            assert "signal" in data
            assert "fs" in data
            assert data["signal"].shape[0] == 12  # 12 leads

    def test_incart_streamer_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            rec, ann = streamer.stream_incart("I01", sampto=500, use_cache=True, fallback_to_synthetic=True)
            assert rec is not None
            assert ann is not None
            assert rec.p_signal.shape[1] == 12

    def test_synthetic_record_wrsamp_caching_robustness(self):
        """Verify synthetic records have valid fmt and can be cached via wfdb.wrsamp without crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            rec, ann = streamer.create_synthetic_record("test_synth", duration_sec=10.0, fs=360.0)
            assert rec.fmt == ["16", "16"]
            assert rec.adc_gain == [200.0, 200.0]
            assert rec.baseline == [0, 0]

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

    def test_incart_caching_mechanism(self):
        """Verify INCART records are written to and loaded from local disk cache."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            rec, ann = streamer.create_synthetic_record("I01", duration_sec=5.0, fs=257.0, num_leads=12)
            db_dir = Path(tmp_dir) / "incartdb"
            db_dir.mkdir(parents=True, exist_ok=True)
            wfdb.wrsamp(
                "I01",
                fs=rec.fs,
                units=rec.units,
                sig_name=rec.sig_name,
                p_signal=rec.p_signal,
                fmt=rec.fmt,
                adc_gain=rec.adc_gain,
                baseline=rec.baseline,
                write_dir=str(db_dir),
            )
            wfdb.wrann(
                "I01",
                "atr",
                sample=ann.sample,
                symbol=ann.symbol,
                write_dir=str(db_dir),
            )
            rec_cached, ann_cached = streamer.stream_incart("I01", use_cache=True)
            assert rec_cached.record_name == "I01"
            assert rec_cached.p_signal.shape == rec.p_signal.shape
            assert len(ann_cached.sample) == len(ann.sample)

