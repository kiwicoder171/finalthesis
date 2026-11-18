"""
Unit and Integration Test Suite for Milestone 2:
Method A (Unified Model Interface, Feature Extractor, and GBDT Classifier).
Adheres strictly to PROJECT.md Contract 2 and ANSI/AAMI EC57 standards.
"""

import os
import tempfile
from typing import Tuple
import numpy as np
import pytest

from src.models.base_classifier import BaseECGClassifier
from src.models.method_a_gbdt import MethodAGBDTClassifier, extract_features


# ============================================================================
# 1. Base Classifier Interface and Input Validation
# ============================================================================

class TestBaseECGClassifierInterface:
    def test_abstract_class_cannot_be_instantiated(self):
        """Verify that BaseECGClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseECGClassifier()

    def test_subclass_must_implement_all_abstract_methods(self):
        """Incomplete subclass without abstract methods raises TypeError."""
        class IncompleteClassifier(BaseECGClassifier):
            def fit(self, X_w, X_r, y, val_data=None):
                pass

        with pytest.raises(TypeError):
            IncompleteClassifier()


class TestInputValidation:
    def test_valid_3d_and_2d_waveforms(self):
        """Validates standard 3D (N, 1, 360) and 2D (N, 360) inputs."""
        w3d = np.zeros((10, 1, 360), dtype=np.float32)
        w2d = np.zeros((10, 360), dtype=np.float32)
        rh = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=int)

        w_out, r_out, y_out = BaseECGClassifier.validate_inputs(w3d, rh, y)
        assert w_out.shape == (10, 1, 360)
        assert r_out.shape == (10, 4)
        assert y_out is not None and len(y_out) == 10

        w_out2, _, _ = BaseECGClassifier.validate_inputs(w2d, rh, y)
        assert w_out2.shape == (10, 1, 360)

    def test_single_beat_1d_input(self):
        """Single beat (360,) and (4,) are promoted to batch size 1."""
        w1d = np.zeros(360, dtype=np.float32)
        r1d = np.zeros(4, dtype=np.float32)
        w_out, r_out, y_out = BaseECGClassifier.validate_inputs(w1d, r1d)
        assert w_out.shape == (1, 1, 360)
        assert r_out.shape == (1, 4)
        assert y_out is None

    def test_waveform_channel_transposition(self):
        """Waveform with shape (N, 360, 1) is automatically transposed to (N, 1, 360)."""
        w_transposed = np.zeros((6, 360, 1), dtype=np.float32)
        rh = np.zeros((6, 4), dtype=np.float32)
        w_out, r_out, _ = BaseECGClassifier.validate_inputs(w_transposed, rh)
        assert w_out.shape == (6, 1, 360)
        assert r_out.shape == (6, 4)

    def test_temporal_length_mismatch_raises_error(self):
        """Waveform with length != 360 raises ValueError."""
        with pytest.raises(ValueError, match="Expected temporal dimension 360"):
            BaseECGClassifier.validate_inputs(np.zeros((5, 300)), np.zeros((5, 4)))

    def test_rhythm_dimension_mismatch_raises_error(self):
        """Rhythm with features != 4 raises ValueError."""
        with pytest.raises(ValueError, match="Expected 4 rhythm features"):
            BaseECGClassifier.validate_inputs(np.zeros((5, 360)), np.zeros((5, 3)))

    def test_sample_count_mismatch_raises_error(self):
        """Waveform count != rhythm count raises ValueError."""
        with pytest.raises(ValueError, match="Sample count mismatch"):
            BaseECGClassifier.validate_inputs(np.zeros((10, 360)), np.zeros((8, 4)))

    def test_nan_and_inf_detection(self):
        """NaN or Inf values raise ValueError."""
        bad_wf = np.zeros((5, 360))
        bad_wf[0, 10] = np.nan
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            BaseECGClassifier.validate_inputs(bad_wf, np.zeros((5, 4)))

        bad_rh = np.zeros((5, 4))
        bad_rh[1, 1] = np.inf
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            BaseECGClassifier.validate_inputs(np.zeros((5, 360)), bad_rh)

    def test_invalid_label_range(self):
        """Labels outside [0, 4] raise ValueError."""
        with pytest.raises(ValueError, match="Labels must be in range"):
            BaseECGClassifier.validate_inputs(
                np.zeros((3, 360)), np.zeros((3, 4)), y=np.array([0, 1, 5])
            )
        with pytest.raises(ValueError, match="Labels must be in range"):
            BaseECGClassifier.validate_inputs(
                np.zeros((3, 360)), np.zeros((3, 4)), y=np.array([-1, 0, 1])
            )

    def test_empty_dataset_raises_when_disallowed(self):
        """Empty inputs raise ValueError when allow_empty=False."""
        with pytest.raises(ValueError, match="Input dataset is empty"):
            BaseECGClassifier.validate_inputs(
                np.zeros((0, 360)), np.zeros((0, 4)), allow_empty=False
            )

    def test_label_count_mismatch_raises_error(self):
        """Label length != waveform length raises ValueError."""
        with pytest.raises(ValueError, match="Sample count mismatch"):
            BaseECGClassifier.validate_inputs(
                np.zeros((5, 360)), np.zeros((5, 4)), y=np.array([0, 1, 2])
            )

    def test_label_type_validation(self):
        """Non-integer convertible labels raise ValueError."""
        with pytest.raises(ValueError, match="Labels must be integer convertible"):
            BaseECGClassifier.validate_inputs(
                np.zeros((2, 360)), np.zeros((2, 4)), y=np.array(["invalid", "labels"])
            )


# ============================================================================
# 2. Feature Extractor Shape Invariance and Numerical Integrity
# ============================================================================

class TestFeatureExtractor:
    def test_shape_invariance_single_and_batch(self):
        """Verify (1, 42) for single beat and (N, 42) for batch."""
        f_single = extract_features(np.random.randn(360), np.array([0.8, 0.8, 0.0, 1.0]))
        assert f_single.shape == (1, 42)
        assert f_single.dtype == np.float32

        f_batch = extract_features(
            np.random.randn(50, 1, 360),
            np.random.uniform(0.5, 1.2, (50, 4)),
        )
        assert f_batch.shape == (50, 42)
        assert f_batch.dtype == np.float32

    def test_batch_vs_single_beat_numerical_equivalence(self):
        """Extracting a beat in a batch vs single yields identical values."""
        X_wf = np.random.randn(5, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.5, 1.2, (5, 4)).astype(np.float32)
        f_batch = extract_features(X_wf, X_rh)
        f_single_0 = extract_features(X_wf[0], X_rh[0])
        np.testing.assert_allclose(f_batch[0:1], f_single_0, atol=1e-5)

    def test_empty_input_produces_empty_output(self):
        """Empty input produces (0, 42) array."""
        res = extract_features(np.zeros((0, 1, 360)), np.zeros((0, 4)))
        assert res.shape == (0, 42)
        assert res.dtype == np.float32

    def test_flatline_and_constant_signals(self):
        """Flatline and constant signals extract without NaNs or Infs."""
        f_flat = extract_features(np.zeros(360), np.zeros(4))
        assert f_flat.shape == (1, 42)
        assert not np.isnan(f_flat).any()
        assert not np.isinf(f_flat).any()

        f_const = extract_features(np.full(360, 3.14), np.ones(4))
        assert f_const.shape == (1, 42)
        assert not np.isnan(f_const).any()
        assert not np.isinf(f_const).any()

    def test_feature_extractor_2d_waveform_input(self):
        """Verify feature extraction supports 2D (N, 360) inputs."""
        X_2d = np.random.randn(8, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.0, (8, 4)).astype(np.float32)
        feats = extract_features(X_2d, X_rh)
        assert feats.shape == (8, 42)
        assert feats.dtype == np.float32

    def test_feature_extractor_extreme_values(self):
        """Signals with high amplitude or small rhythm intervals extract safely."""
        X_wf = (np.random.randn(3, 1, 360) * 1e4).astype(np.float32)
        X_rh = np.array([[1e-6, 1e-6, 0.0, 1.0], [5.0, 5.0, 0.0, 1.0], [0.1, 2.0, -1.9, 0.05]])
        feats = extract_features(X_wf, X_rh)
        assert feats.shape == (3, 42)
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()


# ============================================================================
# 3. Method A GBDT Classifier Lifecycle & Contract 2 Compliance
# ============================================================================

class TestMethodAGBDTClassifier:
    @pytest.fixture
    def synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 5-class synthetic ECG dataset."""
        np.random.seed(42)
        n_samples = 100
        X_wf = np.random.randn(n_samples, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.1, size=(n_samples, 4)).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        return X_wf, X_rh, y

    def test_initialization_defaults(self):
        """Verify default hyperparameter state."""
        clf = MethodAGBDTClassifier()
        assert clf.max_iter == 100
        assert clf.learning_rate == 0.1
        assert clf.class_weight == "balanced"
        assert not clf.is_fitted
        assert clf.model_ is None

    def test_invalid_parameters_raise(self):
        """Invalid hyperparameters raise ValueError."""
        with pytest.raises(ValueError):
            MethodAGBDTClassifier(max_iter=0)
        with pytest.raises(ValueError):
            MethodAGBDTClassifier(learning_rate=-0.01)

    def test_unfitted_predict_raises_runtime_error(self, synthetic_data):
        """Calling predict before fit raises RuntimeError."""
        clf = MethodAGBDTClassifier()
        X_wf, X_rh, _ = synthetic_data
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict(X_wf, X_rh)
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict_proba(X_wf, X_rh)

    def test_fit_and_predict_shapes(self, synthetic_data):
        """Fit completes and predict returns (N,) int with values in {0..4}."""
        clf = MethodAGBDTClassifier(max_iter=20, random_state=42)
        X_wf, X_rh, y = synthetic_data
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted

        preds = clf.predict(X_wf, X_rh)
        assert preds.shape == (len(X_wf),)
        assert np.issubdtype(preds.dtype, np.integer)
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_predict_proba_contract(self, synthetic_data):
        """predict_proba returns (N, 5), float32, rows sum to 1.0."""
        clf = MethodAGBDTClassifier(max_iter=20, random_state=42)
        X_wf, X_rh, y = synthetic_data
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (len(X_wf), 5)
        assert proba.dtype == np.float32
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_matches_argmax_proba(self, synthetic_data):
        """clf.predict matches np.argmax(clf.predict_proba, axis=1)."""
        clf = MethodAGBDTClassifier(max_iter=20, random_state=42)
        X_wf, X_rh, y = synthetic_data
        clf.fit(X_wf, X_rh, y)
        preds = clf.predict(X_wf, X_rh)
        proba = clf.predict_proba(X_wf, X_rh)
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

    def test_checkpoint_save_and_load_round_trip(self, synthetic_data):
        """Model checkpoint save and load preserves exact predictions."""
        clf = MethodAGBDTClassifier(max_iter=20, random_state=42)
        X_wf, X_rh, y = synthetic_data
        clf.fit(X_wf, X_rh, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "models", "nested", "method_a.joblib")
            clf.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            loaded_clf = MethodAGBDTClassifier.load_checkpoint(ckpt_path)
            assert loaded_clf.is_fitted
            assert loaded_clf.max_iter == clf.max_iter

            orig_preds = clf.predict(X_wf, X_rh)
            loaded_preds = loaded_clf.predict(X_wf, X_rh)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

            orig_proba = clf.predict_proba(X_wf, X_rh)
            loaded_proba = loaded_clf.predict_proba(X_wf, X_rh)
            np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-6)

    def test_checkpoint_unfitted_save_raises(self):
        """Saving unfitted model checkpoint raises RuntimeError."""
        clf = MethodAGBDTClassifier()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.joblib")
            with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
                clf.save_checkpoint(ckpt_path)

    def test_checkpoint_nonexistent_file_raises(self):
        """Loading non-existent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MethodAGBDTClassifier.load_checkpoint("/path/does/not/exist.joblib")

    def test_single_class_training_set(self):
        """Fit on single class succeeds and predicts probability 1.0 for that class."""
        X_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_rh = np.ones((20, 4), dtype=np.float32)
        y = np.zeros(20, dtype=int)  # Only class 0

        clf = MethodAGBDTClassifier(max_iter=10, random_state=42)
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (20, 5)
        np.testing.assert_allclose(proba[:, 0], 1.0)
        np.testing.assert_allclose(proba[:, 1:], 0.0)
        preds = clf.predict(X_wf, X_rh)
        np.testing.assert_array_equal(preds, np.zeros(20, dtype=int))

    def test_subset_classes_training_set(self):
        """Fit on classes {0, 2} sets unobserved classes to 0.0 with (N, 5) shape."""
        X_wf = np.random.randn(30, 1, 360).astype(np.float32)
        X_rh = np.ones((30, 4), dtype=np.float32)
        y = np.array([0, 2] * 15)

        clf = MethodAGBDTClassifier(max_iter=10, random_state=42)
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (30, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        # Classes 1, 3, 4 must be 0
        assert np.all(proba[:, 1] == 0.0)
        assert np.all(proba[:, 3] == 0.0)
        assert np.all(proba[:, 4] == 0.0)
        preds = clf.predict(X_wf, X_rh)
        assert set(preds).issubset({0, 2})

    def test_predict_and_proba_empty_input(self, synthetic_data):
        """Empty input to fitted model returns (0,) and (0, 5) arrays."""
        clf = MethodAGBDTClassifier(max_iter=10, random_state=42)
        X_wf, X_rh, y = synthetic_data
        clf.fit(X_wf, X_rh, y)

        p_empty = clf.predict_proba(np.zeros((0, 1, 360)), np.zeros((0, 4)))
        assert p_empty.shape == (0, 5)
        assert p_empty.dtype == np.float32

        pred_empty = clf.predict(np.zeros((0, 1, 360)), np.zeros((0, 4)))
        assert pred_empty.shape == (0,)
        assert np.issubdtype(pred_empty.dtype, np.integer)

    def test_fit_with_val_data(self, synthetic_data):
        """Fitting with validation data completes without error."""
        X_wf, X_rh, y = synthetic_data
        val_wf = X_wf[:20]
        val_rh = X_rh[:20]
        val_y = y[:20]

        clf = MethodAGBDTClassifier(max_iter=20, random_state=42)
        clf.fit(X_wf, X_rh, y, val_data=(val_wf, val_rh, val_y))
        assert clf.is_fitted
        preds = clf.predict(val_wf, val_rh)
        assert len(preds) == 20
