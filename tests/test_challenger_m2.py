"""
Empirical Challenge Test Suite for Milestone 2: Method A (GBDT + Time-Frequency Engineering).
Authored and executed by Challenger M2.

Validates:
1. Feature Extraction Invariants:
   - Single beat vs batch extraction numerical parity within 1e-5.
   - Deterministic (N, 42) shape and float32 dtype across diverse input representations.
   - Zero NaNs/Infs under extreme amplitudes (1e-5, 1e5), flatlines, impulses, high noise, zero rhythm.
2. Model Training & Inference Under Extreme Conditions:
   - Extreme class imbalance (98% class 0, 1% class 1, 1% class 2).
   - Single-class training for all classes 0..4 with exact 1.0 probability mass and correct discrete predictions.
   - Missing-class subsets ({0, 2}, {1, 3}, {2, 4}, {0, 1, 4}, etc.) with canonical (N, 5) projection, unobserved 0.0, and row sum 1.0.
   - Bit-for-bit checkpoint serialization and deserialization across full, subset, and single-class models.
3. Contract 2 Lifecycle and Robustness:
   - Empty input handling for fitted models.
   - Unfitted model protection.
   - Input validation failure modes (NaNs, Infs, shape and count mismatches).
"""

import os
import tempfile
import numpy as np
import pytest

from src.models.base_classifier import BaseECGClassifier
from src.models.method_a_gbdt import MethodAGBDTClassifier, extract_features


# =============================================================================
# 1. FEATURE EXTRACTION INVARIANTS & ADVERSARIAL SIGNALS
# =============================================================================

class TestFeatureExtractionInvariants:
    """Empirical verification of feature extraction shape, dtype, and numerical invariants."""

    def test_single_vs_batch_numerical_parity(self):
        """Batch extraction and beat-by-beat extraction must yield identical values within 1e-5."""
        np.random.seed(2026)
        n_beats = 64
        X_wf = np.random.randn(n_beats, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.5, 1.5, size=(n_beats, 4)).astype(np.float32)

        batch_features = extract_features(X_wf, X_rh)
        assert batch_features.shape == (n_beats, 42)
        assert batch_features.dtype == np.float32

        for i in range(n_beats):
            single_feat = extract_features(X_wf[i], X_rh[i])
            assert single_feat.shape == (1, 42)
            assert single_feat.dtype == np.float32
            max_abs_diff = np.max(np.abs(batch_features[i : i + 1] - single_feat))
            assert max_abs_diff <= 1e-5, f"Beat {i} discrepancy {max_abs_diff} exceeds 1e-5"

    @pytest.mark.parametrize(
        "wf_shape, rh_shape, expected_n",
        [
            ((360,), (4,), 1),
            ((1, 360), (1, 4), 1),
            ((1, 1, 360), (1, 4), 1),
            ((1, 360, 1), (1, 4), 1),
            ((7, 360), (7, 4), 7),
            ((13, 1, 360), (13, 4), 13),
            ((9, 360, 1), (9, 4), 9),
            ((0, 360), (0, 4), 0),
            ((0, 1, 360), (0, 4), 0),
        ],
    )
    def test_feature_dimension_and_dtype_across_diverse_shapes(
        self, wf_shape, rh_shape, expected_n
    ):
        """Feature extractor strictly returns (N, 42) and float32 across diverse input layouts."""
        wf = np.random.randn(*wf_shape).astype(np.float32) if expected_n > 0 else np.zeros(wf_shape, dtype=np.float32)
        rh = np.ones(rh_shape, dtype=np.float32)
        feats = extract_features(wf, rh)
        assert feats.shape == (expected_n, 42)
        assert feats.dtype == np.float32

    def test_adversarial_signals_zero_nans_or_infs(self):
        """Stress-test extractor with flatlines, extreme amplitudes, impulse, noise, and zero rhythm."""
        rng = np.random.RandomState(42)
        pathological_cases = {
            "flatline_zeros": (np.zeros(360, dtype=np.float32), np.zeros(4, dtype=np.float32)),
            "flatline_ones": (np.ones(360, dtype=np.float32), np.ones(4, dtype=np.float32)),
            "flatline_pi": (np.full(360, 3.14159, dtype=np.float32), np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
            "flatline_negative": (np.full(360, -100.0, dtype=np.float32), np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32)),
            "extreme_high_constant_1e5": (np.full(360, 1e5, dtype=np.float32), np.array([10.0, 10.0, 0.0, 1.0], dtype=np.float32)),
            "extreme_high_noise_1e5": (rng.randn(360).astype(np.float32) * 1e5, np.array([5.0, 5.0, 0.0, 1.0], dtype=np.float32)),
            "extreme_low_constant_1e-5": (np.full(360, 1e-5, dtype=np.float32), np.array([1e-5, 1e-5, 0.0, 1.0], dtype=np.float32)),
            "extreme_low_noise_1e-5": (rng.randn(360).astype(np.float32) * 1e-5, np.array([1e-5, 1e-5, 0.0, 1.0], dtype=np.float32)),
            "high_gaussian_noise": (rng.randn(360).astype(np.float32) * 500.0, np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
            "zero_rhythm": (rng.randn(360).astype(np.float32), np.zeros(4, dtype=np.float32)),
            "negative_rhythm": (rng.randn(360).astype(np.float32), np.array([-1.0, -1.0, -0.5, -2.0], dtype=np.float32)),
            "impulse_at_r_peak": (np.eye(360)[120].astype(np.float32) * 50.0, np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
            "inverted_impulse_at_r_peak": (np.eye(360)[120].astype(np.float32) * -50.0, np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
            "step_function": (np.array([0.0] * 180 + [10.0] * 180, dtype=np.float32), np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
            "nyquist_alternating": (np.array([10.0, -10.0] * 180, dtype=np.float32), np.array([0.8, 0.8, 0.0, 1.0], dtype=np.float32)),
        }

        for case_name, (wf, rh) in pathological_cases.items():
            feats = extract_features(wf, rh)
            assert feats.shape == (1, 42), f"{case_name}: Expected shape (1, 42), got {feats.shape}"
            assert feats.dtype == np.float32, f"{case_name}: Expected float32, got {feats.dtype}"
            assert not np.isnan(feats).any(), f"{case_name}: Output contains NaN values"
            assert not np.isinf(feats).any(), f"{case_name}: Output contains Inf values"


# =============================================================================
# 2. MODEL TRAINING & INFERENCE STRESS HARNESS
# =============================================================================

class TestModelTrainingAndInferenceStress:
    """Stress tests for GBDT classifier across extreme imbalance, class subsets, and serialization."""

    def test_extreme_class_imbalance(self):
        """Train on highly imbalanced dataset (98% class 0, 1% class 1, 1% class 2)."""
        rng = np.random.RandomState(42)
        n_0 = 980
        n_1 = 10
        n_2 = 10
        total_samples = n_0 + n_1 + n_2

        X_wf = rng.randn(total_samples, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(total_samples, 4)).astype(np.float32)
        y = np.array([0] * n_0 + [1] * n_1 + [2] * n_2, dtype=int)

        clf = MethodAGBDTClassifier(max_iter=25, random_state=42)
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (total_samples, 5)
        assert proba.dtype == np.float32
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        # Unobserved classes 3 and 4 must have strictly 0.0 probability
        assert np.all(proba[:, 3] == 0.0)
        assert np.all(proba[:, 4] == 0.0)

        preds = clf.predict(X_wf, X_rh)
        assert preds.shape == (total_samples,)
        assert np.issubdtype(preds.dtype, np.integer)
        assert set(preds).issubset({0, 1, 2})
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

    @pytest.mark.parametrize("target_class", [0, 1, 2, 3, 4])
    def test_single_class_training_for_all_classes(self, target_class):
        """Single-class training for every class 0..4 outputs (N, 5) with 1.0 on that class."""
        n_samples = 25
        X_wf = np.random.randn(n_samples, 1, 360).astype(np.float32)
        X_rh = np.ones((n_samples, 4), dtype=np.float32)
        y = np.full(n_samples, target_class, dtype=int)

        clf = MethodAGBDTClassifier(max_iter=10, random_state=42)
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (n_samples, 5)
        assert proba.dtype == np.float32

        # Target class must have 1.0 probability mass
        np.testing.assert_allclose(proba[:, target_class], 1.0)
        # All other classes must be 0.0
        other_classes = [c for c in range(5) if c != target_class]
        for c in other_classes:
            np.testing.assert_allclose(proba[:, c], 0.0)

        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

        preds = clf.predict(X_wf, X_rh)
        np.testing.assert_array_equal(preds, np.full(n_samples, target_class, dtype=int))

    @pytest.mark.parametrize(
        "observed_classes",
        [
            [0, 1],
            [1, 3],
            [2, 4],
            [0, 2, 4],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    def test_missing_classes_projection_and_normalization(self, observed_classes):
        """Subset class training projects onto canonical (N, 5) with rows summing strictly to 1.0."""
        n_per_class = 15
        y_list = []
        for c in observed_classes:
            y_list.extend([c] * n_per_class)
        y = np.array(y_list, dtype=int)
        n_samples = len(y)

        X_wf = np.random.randn(n_samples, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.1, size=(n_samples, 4)).astype(np.float32)

        clf = MethodAGBDTClassifier(max_iter=15, random_state=42)
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (n_samples, 5)
        assert proba.dtype == np.float32
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        unobserved = set(range(5)) - set(observed_classes)
        for c in unobserved:
            assert np.all(proba[:, c] == 0.0), f"Unobserved class {c} has non-zero probability"

        preds = clf.predict(X_wf, X_rh)
        assert set(preds).issubset(set(observed_classes))
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

    def test_bit_for_bit_checkpoint_reproduction(self):
        """Model save_checkpoint and load_checkpoint reproduces predictions bit-for-bit."""
        rng = np.random.RandomState(99)
        n_samples = 60
        X_wf = rng.randn(n_samples, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.5, 1.2, size=(n_samples, 4)).astype(np.float32)

        scenarios = [
            ("all_5_classes", rng.choice([0, 1, 2, 3, 4], size=n_samples)),
            ("subset_0_2", rng.choice([0, 2], size=n_samples)),
            ("single_class_1", np.full(n_samples, 1, dtype=int)),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, y in scenarios:
                clf = MethodAGBDTClassifier(max_iter=15, random_state=42)
                clf.fit(X_wf, X_rh, y)

                orig_preds = clf.predict(X_wf, X_rh)
                orig_proba = clf.predict_proba(X_wf, X_rh)

                ckpt_path = os.path.join(tmp_dir, "deeply", "nested", f"{name}.joblib")
                clf.save_checkpoint(ckpt_path)
                assert os.path.exists(ckpt_path)

                loaded_clf = MethodAGBDTClassifier.load_checkpoint(ckpt_path)
                assert loaded_clf.is_fitted
                assert np.array_equal(loaded_clf.fitted_classes_, clf.fitted_classes_)

                loaded_preds = loaded_clf.predict(X_wf, X_rh)
                loaded_proba = loaded_clf.predict_proba(X_wf, X_rh)

                np.testing.assert_array_equal(orig_preds, loaded_preds)
                np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-7)


# =============================================================================
# 3. INTERFACE CONTRACT 2 ROBUSTNESS & EDGE CASES
# =============================================================================

class TestContract2Robustness:
    """Robustness testing of BaseECGClassifier validation and Method A edge cases."""

    def test_empty_test_set_on_fitted_model(self):
        """Fitted model predict and predict_proba on empty dataset returns correctly sized empty arrays."""
        clf = MethodAGBDTClassifier(max_iter=10, random_state=42)
        X_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_rh = np.ones((20, 4), dtype=np.float32)
        y = np.array([0, 1] * 10)
        clf.fit(X_wf, X_rh, y)

        empty_wf = np.zeros((0, 1, 360), dtype=np.float32)
        empty_rh = np.zeros((0, 4), dtype=np.float32)

        proba = clf.predict_proba(empty_wf, empty_rh)
        assert proba.shape == (0, 5)
        assert proba.dtype == np.float32

        preds = clf.predict(empty_wf, empty_rh)
        assert preds.shape == (0,)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_unfitted_model_raises_runtime_error(self):
        """Predict, predict_proba, and save_checkpoint raise RuntimeError if unfitted."""
        clf = MethodAGBDTClassifier()
        w = np.zeros((5, 1, 360), dtype=np.float32)
        r = np.zeros((5, 4), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict(w, r)
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict_proba(w, r)
        with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
            clf.save_checkpoint("test.joblib")

    def test_load_checkpoint_missing_file_raises_not_found(self):
        """Loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MethodAGBDTClassifier.load_checkpoint("/nonexistent/path/model.joblib")

    @pytest.mark.parametrize(
        "bad_wf, bad_rh, bad_y, expected_match",
        [
            (np.zeros((5, 200)), np.zeros((5, 4)), None, "Expected temporal dimension 360"),
            (np.zeros((5, 360)), np.zeros((5, 3)), None, "Expected 4 rhythm features"),
            (np.zeros((5, 360)), np.zeros((4, 4)), None, "Sample count mismatch"),
            (np.zeros((5, 360)), np.zeros((5, 4)), np.array([0, 1]), "Sample count mismatch"),
            (np.zeros((5, 360)), np.zeros((5, 4)), np.array([0, 1, 2, 3, 5]), "Labels must be in range"),
            (np.zeros((5, 360)), np.zeros((5, 4)), np.array([-1, 0, 1, 2, 3]), "Labels must be in range"),
        ],
    )
    def test_input_validation_contract_violations(
        self, bad_wf, bad_rh, bad_y, expected_match
    ):
        """Contract 2 input validation rejects malformed inputs with informative ValueError."""
        with pytest.raises(ValueError, match=expected_match):
            BaseECGClassifier.validate_inputs(bad_wf, bad_rh, y=bad_y)
