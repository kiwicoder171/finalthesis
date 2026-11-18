"""
Empirical Challenger Test Suite — Iteration 2.
Independent stress-testing of Milestone 1 remediation:
1. Preprocessor filter stability across extreme sampling frequencies and cutoff boundaries.
2. Segmenter numerical invariants: chronological monotonicity, non-negative RR intervals,
   non-negative ratios, boundary padding with allow_padding=True on out-of-domain peaks,
   and empty signal handling.
3. Streamer serialization robustness and cache round-trips.
"""

from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
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
)
from src.data.wfdb_streamer import WFDBStreamer


# =============================================================================
# 1. PREPROCESSOR FILTER STABILITY ACROSS EXTREME FREQUENCIES & BOUNDARIES
# =============================================================================

class TestPreprocessorFilterStabilityAdversarial:
    """Stress testing the Butterworth bandpass filter across boundary conditions."""

    @pytest.mark.parametrize(
        "fs",
        [
            1.0001,
            1.01,
            1.05,
            1.1,
            1.2,
            1.5,
            2.0,
            2.5,
            3.0,
            5.0,
            10.0,
            50.0,
            100.0,
            250.0,
            360.0,
            500.0,
            1000.0,
            5000.0,
        ],
    )
    def test_filter_stability_across_fs_spectrum(self, fs: float):
        """Verify bandpass filter designs stably and executes without crashing across 1.0001 Hz to 5000 Hz."""
        prep = ECGPreprocessor(lowcut=0.5, highcut=45.0)
        # Create a signal of sufficient duration
        n_samples = max(int(fs * 5.0), 30)
        sig = np.sin(2 * np.pi * 0.8 * np.arange(n_samples) / fs) + 0.1 * np.random.randn(n_samples)

        filtered = prep.bandpass_filter(sig, fs=fs)
        assert filtered.shape == sig.shape
        assert np.all(np.isfinite(filtered)), f"Non-finite values produced at fs={fs}"

    @pytest.mark.parametrize(
        "fs,lowcut,highcut",
        [
            (2.0, 0.5, 45.0),    # nyquist = 1.0, highcut > nyquist
            (1.02, 0.5, 45.0),   # nyquist = 0.51, highcut > nyquist
            (1.001, 0.5, 45.0),  # nyquist = 0.5005, highcut > nyquist
            (10.0, 1.0, 5.0),    # highcut == nyquist
            (10.0, 1.0, 6.0),    # highcut > nyquist
            (100.0, 0.1, 49.9),  # highcut near nyquist
        ],
    )
    def test_filter_highcut_clamping_safeguards(self, fs: float, lowcut: float, highcut: float):
        """Verify that highcut >= nyquist is correctly clamped strictly below nyquist and above lowcut."""
        prep = ECGPreprocessor(lowcut=lowcut, highcut=highcut)
        sig = np.random.randn(200).astype(np.float64)
        filtered = prep.bandpass_filter(sig, fs=fs, lowcut=lowcut, highcut=highcut)
        assert filtered.shape == sig.shape
        assert np.all(np.isfinite(filtered))

    @pytest.mark.parametrize(
        "fs,lowcut,highcut,expected_err",
        [
            (0.0, 0.5, 45.0, "Sampling frequency must be positive"),
            (-10.0, 0.5, 45.0, "Sampling frequency must be positive"),
            (100.0, 0.0, 45.0, "Low cutoff frequency must be positive"),
            (100.0, -1.0, 45.0, "Low cutoff frequency must be positive"),
            (1.0, 0.5, 45.0, "too low for lowcut"),       # nyquist == lowcut
            (0.8, 0.5, 45.0, "too low for lowcut"),       # nyquist < lowcut
            (100.0, 20.0, 10.0, "strictly greater than low cutoff"), # high <= low
            (100.0, 20.0, 20.0, "strictly greater than low cutoff"), # high == low
        ],
    )
    def test_filter_invalid_boundary_exceptions(self, fs: float, lowcut: float, highcut: float, expected_err: str):
        """Verify defensive exceptions on mathematically invalid cutoff or frequency combinations."""
        prep = ECGPreprocessor()
        sig = np.ones(100)
        with pytest.raises(ValueError, match=expected_err):
            prep.bandpass_filter(sig, fs=fs, lowcut=lowcut, highcut=highcut)

    @pytest.mark.parametrize(
        "shape",
        [
            (0,),
            (0, 2),
            (2, 0),
            (0, 0),
        ],
    )
    def test_preprocessor_empty_signal_resilience(self, shape: tuple):
        """Verify all preprocessor methods handle 0-sized arrays gracefully without exception."""
        prep = ECGPreprocessor()
        sig = np.zeros(shape, dtype=np.float32)

        # 1. bandpass_filter
        filtered = prep.bandpass_filter(sig, fs=360.0)
        assert filtered.shape == shape

        # 2. detrend
        detrended = prep.detrend(sig)
        assert detrended.shape == shape

        # 3. resample
        resampled = prep.resample(sig, orig_fs=360.0, target_fs=250.0)
        assert resampled.shape == shape

        # 4. z_score_normalize
        normed = prep.z_score_normalize(sig)
        assert normed.shape == shape

        # 5. full preprocess
        preprocessed = prep.preprocess(sig, fs=360.0)
        assert preprocessed.shape == shape

    @pytest.mark.parametrize("sig_len", [1, 2, 3, 5, 10, 15, 27])
    def test_preprocessor_short_signal_padding_limit(self, sig_len: int):
        """Verify signal shorter than filtfilt padlen (min_len = 3*max(len(a), len(b))) falls back safely."""
        prep = ECGPreprocessor(filter_order=4)
        sig = np.linspace(1.0, 5.0, sig_len, dtype=np.float64)
        filtered = prep.bandpass_filter(sig, fs=360.0)
        assert filtered.shape == (sig_len,)
        assert np.all(np.isfinite(filtered))

    def test_resample_indices_comprehensive(self):
        """Verify index scaling handles empty, single-element, and zero values."""
        assert len(resample_indices([], 360.0, 250.0)) == 0
        assert len(resample_indices(np.array([]), 360.0, 250.0)) == 0

        # Same frequency returns integer cast
        orig = np.array([0, 100, 360])
        scaled = resample_indices(orig, 360.0, 360.0)
        np.testing.assert_array_equal(scaled, orig)

        # Downsample 360 -> 180 (factor 0.5)
        scaled_down = resample_indices([0, 100, 360], 360.0, 180.0)
        np.testing.assert_array_equal(scaled_down, [0, 50, 180])

        # Upsample 250 -> 500 (factor 2.0)
        scaled_up = resample_indices([10, 25], 250.0, 500.0)
        np.testing.assert_array_equal(scaled_up, [20, 50])


# =============================================================================
# 2. SEGMENTER NUMERICAL INVARIANTS & BOUNDARY PADDING
# =============================================================================

class TestSegmenterNumericalInvariantsAdversarial:
    """Stress testing monotonicity, non-negativity, out-of-domain peaks, and empty signals."""

    def test_chronological_monotonicity_under_random_permutations(self):
        """Verify annotations provided in arbitrary shuffled order are sorted monotonically."""
        fs = 360.0
        n_samples = 36000  # 100 seconds
        sig = np.random.randn(n_samples).astype(np.float32)

        # Generate 50 random beat locations
        rng = np.random.RandomState(42)
        true_peaks = np.sort(rng.choice(np.arange(1000, n_samples - 1000), size=50, replace=False))
        shuffled_peaks = rng.permutation(true_peaks)
        annotations = [(p, "N") for p in shuffled_peaks]

        res = extract_heartbeats("shuffled", sig, fs, annotations, allow_padding=False)
        extracted_peaks = res["r_peaks"]

        # 1. Monotonicity check
        assert np.all(np.diff(extracted_peaks) >= 0), "Extracted r_peaks must be chronologically non-decreasing"
        # 2. RR intervals non-negativity
        rf = res["rhythm_features"]
        assert np.all(rf[:, 0] >= 0.0), "RR_pre must be non-negative"
        assert np.all(rf[:, 1] >= 0.0), "RR_post must be non-negative"
        assert np.all(rf[:, 3] >= 0.0), "RR_ratio must be non-negative"
        # 3. delta_RR invariant: delta_RR = RR_pre - RR_post
        np.testing.assert_allclose(rf[:, 2], rf[:, 0] - rf[:, 1], atol=1e-5)

    def test_duplicate_r_peaks_invariants(self):
        """Verify identical consecutive R-peaks do not cause divide-by-zero or crashes."""
        fs = 360.0
        sig = np.random.randn(5000).astype(np.float32)
        # Duplicate peaks at 1000 and 2000
        ann = [(1000, "N"), (1000, "V"), (2000, "N"), (2000, "N"), (3000, "N")]
        res = extract_heartbeats("duplicates", sig, fs, ann)

        assert res["beats"].shape[0] == 5
        rf = res["rhythm_features"]
        assert np.all(np.isfinite(rf)), "All rhythm features must be finite even with zero RR intervals"
        assert np.all(rf[:, 0] >= 0.0)
        assert np.all(rf[:, 1] >= 0.0)
        assert np.all(rf[:, 3] >= 0.0)

    def test_compute_rhythm_features_extreme_counts(self):
        """Verify compute_rhythm_features on 0, 1, and 2 beats."""
        # 0 beats
        rf0 = compute_rhythm_features(np.array([], dtype=int), fs=360.0)
        assert rf0.shape == (0, 4)

        # 1 beat
        rf1 = compute_rhythm_features(np.array([500], dtype=int), fs=360.0)
        assert rf1.shape == (1, 4)
        np.testing.assert_allclose(rf1[0], [0.8, 0.8, 0.0, 1.0], atol=1e-5)

        # 2 beats
        rf2 = compute_rhythm_features(np.array([360, 720], dtype=int), fs=360.0)
        assert rf2.shape == (2, 4)
        # dt = 360 / 360 = 1.0 s
        np.testing.assert_allclose(rf2[0], [1.0, 1.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(rf2[1], [1.0, 1.0, 0.0, 1.0], atol=1e-5)

    @pytest.mark.parametrize("allow_padding", [False, True])
    def test_out_of_domain_peaks_pruning(self, allow_padding: bool):
        """Verify negative and beyond-domain peaks are pruned before rhythm calculation."""
        fs = 360.0
        sig_len = 2000
        sig = np.random.randn(sig_len).astype(np.float32)

        ann = [
            (-500, "N"),
            (-1, "N"),
            (500, "N"),
            (1000, "N"),
            (sig_len, "N"),      # Out of bounds: index == sig_len
            (sig_len + 500, "N"),
            (99999, "N"),
        ]

        res = extract_heartbeats("oob_test", sig, fs, ann, allow_padding=allow_padding)
        # Only 500 and 1000 are valid within [0, sig_len)
        assert len(res["r_peaks"]) == 2
        np.testing.assert_array_equal(res["r_peaks"], [500, 1000])

    def test_boundary_padding_on_truncated_peaks(self):
        """Verify allow_padding=True on peaks near edges pads to exactly 360 samples."""
        fs = 360.0
        sig_len = 1000
        sig = np.random.randn(sig_len).astype(np.float32)

        # Peak at 50 (start = -70 < 0) and at 950 (end = 1190 > 1000)
        ann = [(50, "N"), (500, "N"), (950, "N")]

        # Without padding: only peak at 500 is within [120, 1000-240 = 760]
        res_no_pad = extract_heartbeats("no_pad", sig, fs, ann, allow_padding=False)
        assert res_no_pad["beats"].shape == (1, 1, 360)
        assert res_no_pad["r_peaks"][0] == 500

        # With padding: all 3 peaks are extracted and padded to 360 samples
        res_pad = extract_heartbeats("pad", sig, fs, ann, allow_padding=True)
        assert res_pad["beats"].shape == (3, 1, 360)
        np.testing.assert_array_equal(res_pad["r_peaks"], [50, 500, 950])
        assert np.all(np.isfinite(res_pad["beats"]))

    def test_allow_padding_on_ultra_short_signal(self):
        """Verify allow_padding=True when signal length is smaller than pre_samples or window_len."""
        fs = 360.0
        sig = np.random.randn(40).astype(np.float32)  # 40 samples only
        ann = [(20, "N")]

        res = extract_heartbeats("short_padded", sig, fs, ann, allow_padding=True)
        assert res["beats"].shape == (1, 1, 360)
        assert np.all(np.isfinite(res["beats"]))

    @pytest.mark.parametrize(
        "empty_signal",
        [
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((2, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        ],
    )
    def test_empty_signals_contract_1_structure(self, empty_signal: np.ndarray):
        """Verify empty signals return exact Contract 1 structure with 0-length arrays."""
        res = extract_heartbeats("empty_test", empty_signal, fs=360.0, annotations=[(100, "N")])
        assert res["beats"].shape == (0, 1, 360)
        assert res["beats"].dtype == np.float32
        assert res["rhythm_features"].shape == (0, 4)
        assert res["rhythm_features"].dtype == np.float32
        assert res["labels"].shape == (0,)
        assert res["labels"].dtype == int
        assert res["r_peaks"].shape == (0,)
        assert res["r_peaks"].dtype == int


# =============================================================================
# 3. STREAMER SERIALIZATION & CACHE ROUND-TRIPS
# =============================================================================

class TestStreamerSerializationAndCachingAdversarial:
    """Stress testing synthetic record generation, wfdb.wrsamp serialization, and cache hits."""

    @pytest.mark.parametrize("num_leads", [1, 2, 3, 8, 12])
    @pytest.mark.parametrize("fs", [100.0, 250.0, 257.0, 360.0, 500.0])
    def test_synthetic_record_generation_and_wrsamp_roundtrip(self, num_leads: int, fs: float):
        """Verify synthetic record across multiple lead counts and sampling rates round-trips via wfdb.wrsamp."""
        streamer = WFDBStreamer()
        rec, ann = streamer.create_synthetic_record(
            record_name=f"syn_L{num_leads}_F{int(fs)}",
            duration_sec=3.0,
            fs=fs,
            num_leads=num_leads,
            hr_bpm=72.0,
        )

        assert rec.p_signal.shape == (int(3.0 * fs), num_leads)
        assert rec.fs == int(fs)
        assert rec.fmt == ["16"] * num_leads
        assert rec.adc_gain == [200.0] * num_leads
        assert rec.baseline == [0] * num_leads

        with tempfile.TemporaryDirectory() as tmp_dir:
            wfdb.wrsamp(
                rec.record_name,
                fs=rec.fs,
                units=rec.units,
                sig_name=rec.sig_name,
                p_signal=rec.p_signal,
                fmt=rec.fmt,
                adc_gain=rec.adc_gain,
                baseline=rec.baseline,
                write_dir=tmp_dir,
            )
            wfdb.wrann(
                rec.record_name,
                "atr",
                sample=ann.sample,
                symbol=ann.symbol,
                write_dir=tmp_dir,
            )

            # Read back
            rec_read = wfdb.rdrecord(str(Path(tmp_dir) / rec.record_name))
            ann_read = wfdb.rdann(str(Path(tmp_dir) / rec.record_name), "atr")

            assert rec_read.p_signal.shape == rec.p_signal.shape
            assert rec_read.fs == rec.fs
            assert rec_read.sig_name == rec.sig_name
            np.testing.assert_array_equal(ann_read.sample, ann.sample)
            assert ann_read.symbol == ann.symbol

    def test_incart_cache_hit_and_slicing(self):
        """Verify INCART streamer reads from disk cache without network access and supports partial slicing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            streamer = WFDBStreamer(cache_dir=tmp_dir)
            incart_dir = Path(tmp_dir) / "incartdb"
            incart_dir.mkdir(parents=True, exist_ok=True)

            # Generate synthetic 12-lead INCART record (257 Hz)
            rec, ann = streamer.create_synthetic_record(
                record_name="I01",
                duration_sec=10.0,
                fs=257.0,
                num_leads=12,
            )

            wfdb.wrsamp(
                "I01",
                fs=rec.fs,
                units=rec.units,
                sig_name=rec.sig_name,
                p_signal=rec.p_signal,
                fmt=rec.fmt,
                adc_gain=rec.adc_gain,
                baseline=rec.baseline,
                write_dir=str(incart_dir),
            )
            wfdb.wrann(
                "I01",
                "atr",
                sample=ann.sample,
                symbol=ann.symbol,
                write_dir=str(incart_dir),
            )

            # Sliced cache hit: sampfrom=257, sampto=1285 (samples 1 to 5 seconds)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network prohibited on cache hit")):
                with patch("wfdb.rdrecord", wraps=wfdb.rdrecord) as mock_rd:
                    rec_hit, ann_hit = streamer.stream_incart("I01", sampfrom=257, sampto=1285, use_cache=True)
                    assert rec_hit.p_signal.shape == (1028, 12)
                    assert rec_hit.fs == 257
                    assert len(ann_hit.sample) <= len(ann.sample)
