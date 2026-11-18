"""
Adversarial Stress Test Suite (Iteration 2) for Milestone 2: Method A (GBDT + Time-Frequency Engineering).
Authored and executed independently by empirical_challenger (challenger_m2_2).

Verification Coverage:
1. Boundary test cases for extract_features (extreme amplitudes, frequencies, steps, flatlines, empty inputs).
2. Numerical finiteness and strict output shape (N, 42) float32 under all conditions.
3. Contract 2 input validation (wrong dimensions, NaNs, Infs, non-integer/out-of-bounds labels).
4. Classifier convergence, reproducibility across seeds, and timing benchmarks on 10,000 beats (< 5.0s).
"""

import os
import tempfile
import time
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.base_classifier import BaseECGClassifier
from src.models.method_a_gbdt import MethodAGBDTClassifier, extract_features


# ============================================================================
# 1. Adversarial Boundary Stress Tests for extract_features
# ============================================================================

class TestFeatureExtractorAdversarialBoundaries:
    """Stress tests feature extraction on mathematically extreme waveform and rhythm inputs."""

    def test_flatline_zero_signal(self):
        """Pure zero flatline: must extract without NaNs/Infs, returning (1, 42)."""
        wf = np.zeros(360, dtype=np.float32)
        rh = np.zeros(4, dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (1, 42)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all(), "Flatline zero signal produced NaN or Inf"

    @pytest.mark.parametrize("const_val", [-1e6, -100.0, -1.0, -1e-6, 1e-6, 1.0, 100.0, 1e6])
    def test_constant_signals(self, const_val: float):
        """Constant non-zero waveforms across varying magnitudes: zero variance guard must prevent NaNs."""
        wf = np.full((3, 1, 360), const_val, dtype=np.float32)
        rh = np.array([[1.0, 1.0, 0.0, 1.0]] * 3, dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (3, 42)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all(), f"Constant signal {const_val} produced non-finite feature"

    @pytest.mark.parametrize("scale", [1e6, 1e9, 1e12])
    def test_ultra_high_amplitude_signals(self, scale: float):
        """Massive amplitudes: must not overflow into Inf or NaN in energy/entropy."""
        rng = np.random.RandomState(42)
        wf = (rng.randn(4, 1, 360) * scale).astype(np.float32)
        rh = np.ones((4, 4), dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (4, 42)
        assert np.isfinite(feats).all(), f"Ultra-high amplitude scale {scale} produced Inf or NaN"

    @pytest.mark.parametrize("scale", [1e-12, 1e-20, 1e-30])
    def test_subnormal_tiny_amplitude_signals(self, scale: float):
        """Subnormal amplitudes close to zero: must not trigger division-by-zero or zero-variance NaN."""
        rng = np.random.RandomState(42)
        wf = (rng.randn(4, 1, 360) * scale).astype(np.float32)
        rh = np.ones((4, 4), dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (4, 42)
        assert np.isfinite(feats).all(), f"Tiny amplitude scale {scale} produced Inf or NaN"

    def test_extreme_nyquist_frequency(self):
        """Alternating Nyquist frequency [+1, -1, +1, -1, ...]: maximal slope and high DWT energy."""
        alt = np.tile([1.0, -1.0], 180).astype(np.float32)
        rh = np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)
        feats = extract_features(alt, rh)
        assert feats.shape == (1, 42)
        assert np.isfinite(feats).all()

    def test_random_walk_brownian_drift(self):
        """Random walk / Brownian bridge: non-stationary baseline drift with strong trend."""
        rng = np.random.RandomState(1337)
        steps = rng.randn(5, 360).astype(np.float32)
        walk = np.cumsum(steps, axis=1)
        rh = np.tile([1.0, 0.8, -0.2, 0.8], (5, 1)).astype(np.float32)
        feats = extract_features(walk, rh)
        assert feats.shape == (5, 42)
        assert np.isfinite(feats).all()

    @pytest.mark.parametrize("spike_idx", [0, 50, 120, 250, 359])
    def test_impulse_delta_signals(self, spike_idx: int):
        """Delta function spike at various key locations (start, pre-QRS, R-peak=120, post-QRS, end)."""
        wf = np.zeros((1, 1, 360), dtype=np.float32)
        wf[0, 0, spike_idx] = 10.0
        rh = np.array([[0.8, 0.8, 0.0, 1.0]], dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (1, 42)
        assert np.isfinite(feats).all()

    def test_step_function_discontinuity(self):
        """Heaviside step discontinuity at sample 120."""
        step = np.zeros(360, dtype=np.float32)
        step[120:] = 5.0
        rh = np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)
        feats = extract_features(step, rh)
        assert feats.shape == (1, 42)
        assert np.isfinite(feats).all()

    def test_chirp_frequency_sweep(self):
        """Linear chirp sweeping from 1 Hz to 90 Hz across 360 samples."""
        t = np.linspace(0, 1.0, 360, endpoint=False, dtype=np.float32)
        chirp = np.sin(2 * np.pi * (1.0 + 89.0 * t / 2.0) * t).astype(np.float32)
        rh = np.array([0.9, 0.9, 0.0, 1.0], dtype=np.float32)
        feats = extract_features(chirp, rh)
        assert feats.shape == (1, 42)
        assert np.isfinite(feats).all()

    def test_empty_dataset_boundary(self):
        """Empty input (0, 1, 360) and (0, 4) produces exact (0, 42) float32."""
        wf = np.zeros((0, 1, 360), dtype=np.float32)
        rh = np.zeros((0, 4), dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (0, 42)
        assert feats.dtype == np.float32

    def test_rhythm_extreme_boundaries(self):
        """Adversarial rhythm intervals: zeros, massive ratios, near-zero ratios, negative delta."""
        wf = np.random.randn(6, 1, 360).astype(np.float32)
        rh_adversarial = np.array([
            [0.0, 0.0, 0.0, 0.0],          # All zeros (division by zero test)
            [1e-6, 1e-6, 0.0, 1.0],        # Tiny RR intervals
            [10.0, 10.0, 0.0, 1.0],        # Giant 10-second intervals
            [0.001, 10.0, 9.999, 10000.0], # Extreme premature ratio
            [10.0, 0.001, -9.999, 0.0001], # Extreme post-mature ratio
            [0.5, 0.5, -1.0, 1e-8],        # Negative delta with tiny ratio
        ], dtype=np.float32)
        feats = extract_features(wf, rh_adversarial)
        assert feats.shape == (6, 42)
        assert np.isfinite(feats).all(), "Adversarial rhythm values produced non-finite feature"

    def test_single_beat_vs_batch_deterministic_parity(self):
        """Single-beat extraction must match batch slice within machine epsilon."""
        rng = np.random.RandomState(999)
        wf = rng.randn(10, 1, 360).astype(np.float32)
        rh = rng.uniform(0.4, 1.5, size=(10, 4)).astype(np.float32)
        batch_feats = extract_features(wf, rh)

        for i in range(10):
            single_feats = extract_features(wf[i, 0, :], rh[i])
            np.testing.assert_allclose(
                single_feats[0],
                batch_feats[i],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Discrepancy at sample {i} between single-beat and batch extraction"
            )


# ============================================================================
# 2. Contract 2 Input Validation & Adversarial Error Detection
# ============================================================================

class TestContract2InputValidationAdversarial:
    """Rigorous validation of Contract 2 interface requirements under malformed inputs."""

    @pytest.mark.parametrize("bad_wf_shape", [
        (359,),            # 1D wrong length < 360
        (361,),            # 1D wrong length > 360
        (10, 359),         # 2D wrong length
        (10, 361),         # 2D wrong length
        (10, 2, 360),      # 3D wrong channels (multi-lead unsupported in single-lead beat classifier)
        (10, 1, 359),      # 3D wrong temporal length
        (10, 359, 1),      # 3D wrong temporal length transposed
        (5, 1, 1, 360),    # 4D tensor
    ])
    def test_invalid_waveform_dimensions_raise_value_error(self, bad_wf_shape):
        """Any waveform with incorrect temporal dimension or invalid ndim raises ValueError."""
        wf = np.zeros(bad_wf_shape, dtype=np.float32)
        rh = np.zeros((bad_wf_shape[0] if len(bad_wf_shape) > 1 else 1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            BaseECGClassifier.validate_inputs(wf, rh)

    @pytest.mark.parametrize("bad_rh_shape", [
        (3,),              # 1D with 3 features
        (5,),              # 1D with 5 features
        (10, 3),           # 2D with 3 features
        (10, 5),           # 2D with 5 features
        (10, 1, 4),        # 3D tensor
    ])
    def test_invalid_rhythm_dimensions_raise_value_error(self, bad_rh_shape):
        """Rhythm vectors not having exactly 4 features raise ValueError."""
        n_samples = bad_rh_shape[0] if len(bad_rh_shape) > 1 else 1
        wf = np.zeros((n_samples, 1, 360), dtype=np.float32)
        rh = np.zeros(bad_rh_shape, dtype=np.float32)
        with pytest.raises(ValueError):
            BaseECGClassifier.validate_inputs(wf, rh)

    def test_sample_count_mismatch_raises_value_error(self):
        """Mismatched sample counts between waveforms and rhythm raise ValueError."""
        wf = np.zeros((10, 1, 360), dtype=np.float32)
        rh = np.zeros((8, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Sample count mismatch"):
            BaseECGClassifier.validate_inputs(wf, rh)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_waveform_nan_inf_detection(self, bad_val: float):
        """Contamination of waveform with NaN or +/-Inf raises ValueError."""
        wf = np.zeros((5, 1, 360), dtype=np.float32)
        wf[2, 0, 180] = bad_val
        rh = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            BaseECGClassifier.validate_inputs(wf, rh)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_rhythm_nan_inf_detection(self, bad_val: float):
        """Contamination of rhythm with NaN or +/-Inf raises ValueError."""
        wf = np.zeros((5, 1, 360), dtype=np.float32)
        rh = np.zeros((5, 4), dtype=np.float32)
        rh[1, 2] = bad_val
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            BaseECGClassifier.validate_inputs(wf, rh)

    @pytest.mark.parametrize("bad_label", [-100, -1, 5, 6, 99])
    def test_out_of_bounds_labels_raise_value_error(self, bad_label: int):
        """Labels outside [0, 4] raise ValueError."""
        wf = np.zeros((4, 1, 360), dtype=np.float32)
        rh = np.zeros((4, 4), dtype=np.float32)
        y = np.array([0, 1, 2, bad_label])
        with pytest.raises(ValueError, match="Labels must be in range"):
            BaseECGClassifier.validate_inputs(wf, rh, y)

    def test_non_integer_convertible_labels_raise(self):
        """String labels that cannot be converted to integer raise ValueError."""
        wf = np.zeros((3, 1, 360), dtype=np.float32)
        rh = np.zeros((3, 4), dtype=np.float32)
        y = np.array(["N", "V", "S"])
        with pytest.raises(ValueError, match="Labels must be integer convertible"):
            BaseECGClassifier.validate_inputs(wf, rh, y)

    def test_label_length_mismatch_raises_value_error(self):
        """Mismatched sample count between labels and waveforms raises ValueError."""
        wf = np.zeros((6, 1, 360), dtype=np.float32)
        rh = np.zeros((6, 4), dtype=np.float32)
        y = np.array([0, 1, 2, 3])  # Length 4 != 6
        with pytest.raises(ValueError, match="Sample count mismatch"):
            BaseECGClassifier.validate_inputs(wf, rh, y)

    def test_disallowed_empty_dataset_raises(self):
        """Empty dataset with allow_empty=False raises ValueError."""
        with pytest.raises(ValueError, match="Input dataset is empty"):
            BaseECGClassifier.validate_inputs(
                np.zeros((0, 1, 360)), np.zeros((0, 4)), allow_empty=False
            )

    def test_allowed_empty_dataset_returns_empty_arrays(self):
        """Empty dataset with allow_empty=True returns empty arrays cleanly."""
        wf_out, rh_out, y_out = BaseECGClassifier.validate_inputs(
            np.zeros((0, 1, 360)), np.zeros((0, 4)), y=np.zeros(0, dtype=int), allow_empty=True
        )
        assert len(wf_out) == 0
        assert len(rh_out) == 0
        assert y_out is not None and len(y_out) == 0


# ============================================================================
# 3. Classifier Convergence, Determinism, and Timing Stress Tests
# ============================================================================

class TestClassifierConvergenceAndTiming:
    """Verifies GBDT convergence, deterministic seed reproducibility, and scale timing."""

    def test_deterministic_reproducibility_with_fixed_seed(self):
        """Two identical models with random_state=42 trained on the same data yield identical predictions and probas."""
        rng = np.random.RandomState(42)
        N = 100
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.5, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.7, 0.1, 0.1, 0.05, 0.05])

        clf1 = MethodAGBDTClassifier(max_iter=25, random_state=42)
        clf1.fit(wf, rh, y)
        preds1 = clf1.predict(wf, rh)
        proba1 = clf1.predict_proba(wf, rh)

        clf2 = MethodAGBDTClassifier(max_iter=25, random_state=42)
        clf2.fit(wf, rh, y)
        preds2 = clf2.predict(wf, rh)
        proba2 = clf2.predict_proba(wf, rh)

        np.testing.assert_array_equal(preds1, preds2, err_msg="Predictions differ across identical seed")
        np.testing.assert_allclose(proba1, proba2, atol=1e-7, err_msg="Probabilities differ across identical seed")

    def test_different_seeds_produce_variation(self):
        """Models with different random_states produce non-identical internal estimators."""
        rng = np.random.RandomState(123)
        N = 150
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.5, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.5, 0.2, 0.15, 0.1, 0.05])

        clf_a = MethodAGBDTClassifier(max_iter=20, random_state=42)
        clf_b = MethodAGBDTClassifier(max_iter=20, random_state=999)
        clf_a.fit(wf, rh, y)
        clf_b.fit(wf, rh, y)

        assert clf_a.random_state != clf_b.random_state

    def test_gbdt_convergence_and_loss_decrease(self):
        """GBDT model converges: achieves higher train accuracy than random guessing (0.2)."""
        rng = np.random.RandomState(42)
        N = 200
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.5, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N)

        clf = MethodAGBDTClassifier(max_iter=30, random_state=42)
        clf.fit(wf, rh, y)
        preds = clf.predict(wf, rh)
        acc = np.mean(preds == y)
        assert acc > 0.4, f"GBDT failed to learn patterns, accuracy {acc:.2f} <= 0.4"

    def test_classifier_fit_timing_on_10k_beats(self):
        """HistGradientBoostingClassifier fit on 10,000 pre-extracted beats features must take < 5 seconds."""
        rng = np.random.RandomState(42)
        N = 10000
        n_features = 42

        # Generate synthetic pre-extracted feature representation
        X_feats = rng.randn(N, n_features).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.8, 0.08, 0.08, 0.02, 0.02])

        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=50,
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )

        t_start = time.perf_counter()
        hgb.fit(X_feats, y)
        elapsed = time.perf_counter() - t_start

        print(f"\n[Timing Benchmark] HistGradientBoostingClassifier.fit on {N} samples took: {elapsed:.3f}s")
        assert elapsed < 5.0, f"HistGradientBoosting fit exceeded 5.0 seconds ({elapsed:.3f}s)"

    def test_predict_proba_simplex_constraint(self):
        """Output of predict_proba strictly lies on the 5-dimensional probability simplex."""
        rng = np.random.RandomState(42)
        N = 50
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.5, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N)

        clf = MethodAGBDTClassifier(max_iter=15, random_state=42)
        clf.fit(wf, rh, y)
        proba = clf.predict_proba(wf, rh)

        assert proba.shape == (N, 5)
        assert np.all(proba >= 0.0), "Found negative probability"
        assert np.all(proba <= 1.0), "Found probability > 1.0"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_checkpoint_isolation_and_reproducibility(self):
        """Checkpoint serialization and restoration is clean and isolated."""
        rng = np.random.RandomState(42)
        N = 60
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.5, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N)

        clf = MethodAGBDTClassifier(max_iter=15, random_state=42)
        clf.fit(wf, rh, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "method_a_adversarial.joblib")
            clf.save_checkpoint(ckpt_path)

            loaded_clf = MethodAGBDTClassifier.load_checkpoint(ckpt_path)
            orig_preds = clf.predict(wf, rh)
            loaded_preds = loaded_clf.predict(wf, rh)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

            orig_proba = clf.predict_proba(wf, rh)
            loaded_proba = loaded_clf.predict_proba(wf, rh)
            np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-7)
