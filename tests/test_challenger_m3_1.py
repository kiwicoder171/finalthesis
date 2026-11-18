"""
Adversarial Challenger Test Suite for Milestone 3 (Method B: Deep 1D-ResNet with Rhythm Fusion).
Adheres strictly to PROJECT.md Contract 2 and ANSI/AAMI EC57 standards.

Tests empirical robustness under:
1. Extreme Waveforms: huge amplitudes, flatlines, high-frequency Gaussian noise, severe baseline wander (0.1 Hz).
2. Boundary & Corner Cases: single beat shapes (360,), small batches (N=1, 2, 3, 65), extreme rhythm intervals.
3. Class-Subset Behavior: fit on {0, 1}, {2, 3, 4}, {1, 4}, and single classes {0}, {3}.
4. Input Validation & Focal Loss numerical stability.
5. Checkpoint serialization fidelity.
"""

import math
import os
import tempfile
from typing import List, Tuple
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.base_classifier import BaseECGClassifier
from src.models.method_b_resnet import (
    ECGResNet1D,
    MethodBResNetClassifier,
    MultiClassFocalLoss,
    ResidualBlock1D,
)


# ============================================================================
# 1. EXTREME WAVEFORMS & SIGNAL DISTORTIONS
# ============================================================================

class TestExtremeWaveforms:
    """Stress tests Method B against pathological, distorted, and extreme ECG waveforms."""

    @pytest.fixture
    def fitted_model(self) -> MethodBResNetClassifier:
        """Provide a lightweight fitted classifier for inference testing."""
        rng = np.random.RandomState(42)
        n = 50
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)
        return clf

    def test_huge_amplitude_spikes(self, fitted_model: MethodBResNetClassifier):
        """Verify model handles extreme amplitude spikes (up to 1e6) without NaN/Inf."""
        N = 10
        base_wf = np.random.randn(N, 1, 360).astype(np.float32)
        # Inject massive positive and negative spikes
        base_wf[0, 0, 180] = 1e5
        base_wf[1, 0, 120] = -1e5
        base_wf[2, 0, :] = 1e6
        base_wf[3, 0, :] = -1e6

        X_rh = np.ones((N, 4), dtype=np.float32)

        proba = fitted_model.predict_proba(base_wf, X_rh)
        preds = fitted_model.predict(base_wf, X_rh)

        assert not np.isnan(proba).any(), "Found NaN in predict_proba under huge amplitude spikes"
        assert not np.isinf(proba).any(), "Found Inf in predict_proba under huge amplitude spikes"
        assert (proba >= 0.0).all() and (proba <= 1.0).all(), "Probabilities not bounded in [0, 1]"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        assert len(preds) == N
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()

    def test_complete_flatline_waveforms(self, fitted_model: MethodBResNetClassifier):
        """Verify model handles zero-flatline and constant-voltage flatline signals."""
        N = 6
        flatlines = np.zeros((N, 1, 360), dtype=np.float32)
        flatlines[0, 0, :] = 0.0       # Complete zero signal
        flatlines[1, 0, :] = 5.0       # Constant DC offset +5V
        flatlines[2, 0, :] = -5.0      # Constant DC offset -5V
        flatlines[3, 0, :] = 100.0     # Constant DC offset +100V
        flatlines[4, 0, :] = 1e-4      # Micro-volt constant signal
        flatlines[5, 0, :] = -1e-4

        X_rh = np.full((N, 4), 0.8, dtype=np.float32)

        proba = fitted_model.predict_proba(flatlines, X_rh)
        preds = fitted_model.predict(flatlines, X_rh)

        assert not np.isnan(proba).any(), "Flatline caused NaN in predict_proba"
        assert not np.isinf(proba).any(), "Flatline caused Inf in predict_proba"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()

    def test_high_frequency_gaussian_noise(self, fitted_model: MethodBResNetClassifier):
        """Verify model stability under heavy Gaussian noise (sigma = 10, 100, 500)."""
        rng = np.random.RandomState(123)
        N = 6
        sigmas = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
        noisy_wf = np.empty((N, 1, 360), dtype=np.float32)
        for i, s in enumerate(sigmas):
            noisy_wf[i, 0, :] = rng.normal(loc=0.0, scale=s, size=360).astype(np.float32)

        X_rh = rng.uniform(0.5, 1.5, size=(N, 4)).astype(np.float32)

        proba = fitted_model.predict_proba(noisy_wf, X_rh)
        preds = fitted_model.predict(noisy_wf, X_rh)

        assert not np.isnan(proba).any()
        assert not np.isinf(proba).any()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()

    def test_severe_baseline_wander(self, fitted_model: MethodBResNetClassifier):
        """
        Verify model stability under severe low-frequency baseline wander (0.05 Hz and 0.1 Hz)
        with massive amplitude sinusoidal modulation (+/- 10 mV, 50 mV).
        """
        fs = 360.0
        t = np.arange(360) / fs
        N = 4
        wander_wf = np.empty((N, 1, 360), dtype=np.float32)

        # 0.1 Hz wander with 10 mV amplitude
        wander_wf[0, 0, :] = (10.0 * np.sin(2 * np.pi * 0.1 * t)).astype(np.float32)
        # 0.05 Hz wander with 20 mV amplitude
        wander_wf[1, 0, :] = (20.0 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)
        # 0.1 Hz wander with 50 mV amplitude + Gaussian noise
        wander_wf[2, 0, :] = (50.0 * np.cos(2 * np.pi * 0.1 * t) + np.random.randn(360)).astype(np.float32)
        # Linear drift across window (from -25.0 to +25.0)
        wander_wf[3, 0, :] = np.linspace(-25.0, 25.0, 360, dtype=np.float32)

        X_rh = np.full((N, 4), 0.75, dtype=np.float32)

        proba = fitted_model.predict_proba(wander_wf, X_rh)
        preds = fitted_model.predict(wander_wf, X_rh)

        assert not np.isnan(proba).any()
        assert not np.isinf(proba).any()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()

    def test_inverted_polarity_waveforms(self, fitted_model: MethodBResNetClassifier):
        """Verify model handles inverted leads / inverted polarity (-1.0 * signal)."""
        rng = np.random.RandomState(999)
        wf = rng.randn(5, 1, 360).astype(np.float32)
        inverted_wf = -1.0 * wf
        rh = rng.uniform(0.6, 1.2, size=(5, 4)).astype(np.float32)

        proba = fitted_model.predict_proba(inverted_wf, rh)
        preds = fitted_model.predict(inverted_wf, rh)

        assert not np.isnan(proba).any()
        assert not np.isinf(proba).any()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert len(preds) == 5

    def test_training_stability_with_pathological_signals(self):
        """
        Verify that training does not diverge or produce NaN weights even when
        the training set contains flatlines, massive noise, and baseline wander.
        """
        rng = np.random.RandomState(42)
        n = 40
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        # Corrupt half the samples with extreme signals
        X_wf[0:5] = 0.0                                          # Flatline
        X_wf[5:10] = 100.0 * rng.randn(5, 1, 360)               # Huge noise
        X_wf[10:15] = 50.0 * np.sin(np.linspace(0, 1, 360))     # Huge wander

        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=3, batch_size=16, random_state=42)
        clf.fit(X_wf, X_rh, y)

        assert clf.is_fitted
        # Verify model parameters are completely finite (no NaNs)
        for param in clf.model_.parameters():
            assert torch.isfinite(param).all(), "Model weight exploded to NaN or Inf during training"


# ============================================================================
# 2. BOUNDARY & CORNER CASES (SHAPES, BATCH SIZES, RHYTHM INTERVALS)
# ============================================================================

class TestBoundaryAndCornerCases:
    """Stress tests boundary input dimensions, small batches, and extreme rhythm intervals."""

    @pytest.fixture
    def fitted_model(self) -> MethodBResNetClassifier:
        rng = np.random.RandomState(42)
        n = 40
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)
        return clf

    def test_single_sample_shape_variations_equivalence(self, fitted_model: MethodBResNetClassifier):
        """
        Verify identical predictions for single beat represented in 4 valid shapes:
        - (360,) and (4,)
        - (1, 360) and (1, 4)
        - (1, 1, 360) and (1, 4)
        - (1, 360, 1) and (1, 4)
        """
        rng = np.random.RandomState(77)
        wf_1d = rng.randn(360).astype(np.float32)
        rh_1d = rng.uniform(0.6, 1.2, size=4).astype(np.float32)

        wf_2d = wf_1d.reshape(1, 360)
        rh_2d = rh_1d.reshape(1, 4)

        wf_3d_channel_first = wf_1d.reshape(1, 1, 360)
        wf_3d_channel_last = wf_1d.reshape(1, 360, 1)

        p1 = fitted_model.predict_proba(wf_1d, rh_1d)
        p2 = fitted_model.predict_proba(wf_2d, rh_2d)
        p3 = fitted_model.predict_proba(wf_3d_channel_first, rh_2d)
        p4 = fitted_model.predict_proba(wf_3d_channel_last, rh_2d)

        assert p1.shape == (1, 5)
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        np.testing.assert_allclose(p1, p3, atol=1e-6)
        np.testing.assert_allclose(p1, p4, atol=1e-6)

        pred1 = fitted_model.predict(wf_1d, rh_1d)
        pred2 = fitted_model.predict(wf_2d, rh_2d)
        assert pred1.shape == (1,)
        assert pred1[0] == pred2[0]

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 7, 13])
    def test_small_batch_inference_invariants(self, fitted_model: MethodBResNetClassifier, N: int):
        """Verify inference succeeds and outputs match shape (N, 5) and (N,) for small batches."""
        rng = np.random.RandomState(N)
        wf = rng.randn(N, 1, 360).astype(np.float32)
        rh = rng.uniform(0.6, 1.2, size=(N, 4)).astype(np.float32)

        proba = fitted_model.predict_proba(wf, rh)
        preds = fitted_model.predict(wf, rh)

        assert proba.shape == (N, 5)
        assert preds.shape == (N,)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()

    def test_small_dataset_training_batch_1_remainder(self):
        """
        Verify training does NOT crash when dataset size leaves a remainder batch of size 1.
        e.g., N=65 with batch_size=64 leaves a batch of size 1.
        """
        rng = np.random.RandomState(42)
        N = 65
        X_wf = rng.randn(N, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(N, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=N)

        clf = MethodBResNetClassifier(epochs=2, batch_size=64, random_state=42)
        # Must train without BatchNorm1d crashing on batch size 1
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted
        preds = clf.predict(X_wf[:5], X_rh[:5])
        assert len(preds) == 5

    def test_minimal_dataset_training_n2_and_n3(self):
        """Verify training on minimal datasets (N=2 and N=3) succeeds cleanly."""
        rng = np.random.RandomState(42)
        # N=2 with classes 0 and 1
        X_wf_2 = rng.randn(2, 1, 360).astype(np.float32)
        X_rh_2 = rng.uniform(0.6, 1.2, size=(2, 4)).astype(np.float32)
        y_2 = np.array([0, 1])

        clf_2 = MethodBResNetClassifier(epochs=2, batch_size=2, random_state=42)
        clf_2.fit(X_wf_2, X_rh_2, y_2)
        assert clf_2.is_fitted
        proba_2 = clf_2.predict_proba(X_wf_2, X_rh_2)
        assert proba_2.shape == (2, 5)
        np.testing.assert_allclose(proba_2.sum(axis=1), 1.0, atol=1e-5)

        # N=3 with classes 0, 1, 2 and batch_size=2 (creates batches of size 2 and 1)
        X_wf_3 = rng.randn(3, 1, 360).astype(np.float32)
        X_rh_3 = rng.uniform(0.6, 1.2, size=(3, 4)).astype(np.float32)
        y_3 = np.array([0, 1, 2])

        clf_3 = MethodBResNetClassifier(epochs=2, batch_size=2, random_state=42)
        clf_3.fit(X_wf_3, X_rh_3, y_3)
        assert clf_3.is_fitted
        preds_3 = clf_3.predict(X_wf_3, X_rh_3)
        assert len(preds_3) == 3

    def test_extreme_rhythm_intervals(self, fitted_model: MethodBResNetClassifier):
        """
        Stress test rhythm feature MLP with extreme values:
        - Negative RR intervals (corrupted detector / artifacts)
        - Zero RR intervals (simultaneous or zero-detected peaks)
        - Extremely large RR intervals (bradycardia / asystole / 10s intervals)
        - Extreme RR ratios
        """
        N = 7
        X_wf = np.random.randn(N, 1, 360).astype(np.float32)
        X_rh = np.empty((N, 4), dtype=np.float32)

        X_rh[0] = [-500.0, -200.0, -300.0, -1.0]    # All negative
        X_rh[1] = [0.0, 0.0, 0.0, 0.0]               # All zero
        X_rh[2] = [10000.0, 25000.0, 15000.0, 2.5]   # Massive values (seconds to hours)
        X_rh[3] = [0.001, 10000.0, 9999.999, 1e7]    # Massive ratio
        X_rh[4] = [10000.0, 0.001, -9999.999, 1e-7]  # Tiny ratio
        X_rh[5] = [1e4, 1e4, 0.0, 1.0]
        X_rh[6] = [-1e3, 1e3, 2e3, -1.0]

        proba = fitted_model.predict_proba(X_wf, X_rh)
        preds = fitted_model.predict(X_wf, X_rh)

        assert not np.isnan(proba).any(), "Extreme rhythm features generated NaN in predict_proba"
        assert not np.isinf(proba).any(), "Extreme rhythm features generated Inf in predict_proba"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.isin(preds, [0, 1, 2, 3, 4]).all()


# ============================================================================
# 3. CLASS-SUBSET AND SINGLE-CLASS BEHAVIOR
# ============================================================================

class TestClassSubsetBehavior:
    """Stress tests classifier behavior when trained on strict subsets of AAMI classes."""

    def test_fit_subset_0_1(self):
        """When trained on {0, 1}, predictions must be in {0, 1} and proba[:, 2:5] == 0."""
        rng = np.random.RandomState(42)
        n = 40
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)

        preds = clf.predict(X_wf, X_rh)
        proba = clf.predict_proba(X_wf, X_rh)

        assert np.isin(preds, [0, 1]).all(), f"Predictions contained unobserved classes: {np.unique(preds)}"
        np.testing.assert_allclose(proba[:, [2, 3, 4]], 0.0, atol=1e-7)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_subset_2_3_4(self):
        """When trained on {2, 3, 4}, predictions must be in {2, 3, 4} and proba[:, [0, 1]] == 0."""
        rng = np.random.RandomState(42)
        n = 40
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)

        preds = clf.predict(X_wf, X_rh)
        proba = clf.predict_proba(X_wf, X_rh)

        assert np.isin(preds, [2, 3, 4]).all(), f"Predictions contained unobserved classes: {np.unique(preds)}"
        np.testing.assert_allclose(proba[:, [0, 1]], 0.0, atol=1e-7)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_subset_noncontiguous_1_4(self):
        """When trained on non-contiguous {1, 4}, predictions must be in {1, 4} and proba[:, [0, 2, 3]] == 0."""
        rng = np.random.RandomState(42)
        n = 30
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([1, 4], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)

        preds = clf.predict(X_wf, X_rh)
        proba = clf.predict_proba(X_wf, X_rh)

        assert np.isin(preds, [1, 4]).all()
        np.testing.assert_allclose(proba[:, [0, 2, 3]], 0.0, atol=1e-7)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("target_class", [0, 1, 2, 3, 4])
    def test_fit_single_class_deterministic_behavior(self, target_class: int):
        """
        When trained on a single class, predict() must return that exact class
        and predict_proba() must assign 1.0 to that class column and 0.0 to all others.
        """
        rng = np.random.RandomState(42)
        n = 20
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = np.full(n, target_class, dtype=int)

        clf = MethodBResNetClassifier(epochs=2, random_state=42)
        clf.fit(X_wf, X_rh, y)

        assert clf.is_fitted
        assert list(clf.fitted_classes_) == [target_class]

        test_n = 15
        X_test_wf = rng.randn(test_n, 1, 360).astype(np.float32)
        X_test_rh = rng.uniform(0.6, 1.2, size=(test_n, 4)).astype(np.float32)

        preds = clf.predict(X_test_wf, X_test_rh)
        proba = clf.predict_proba(X_test_wf, X_test_rh)

        assert (preds == target_class).all(), f"Expected all predictions to be {target_class}, got {preds}"
        assert (proba[:, target_class] == 1.0).all()
        other_classes = [c for c in range(5) if c != target_class]
        assert (proba[:, other_classes] == 0.0).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)


# ============================================================================
# 4. INPUT VALIDATION & NUMERICAL STABILITY (FOCAL LOSS & CHECKPOINTS)
# ============================================================================

class TestInputValidationAndNumericalStability:
    """Stress tests input rejection contracts, focal loss numerical stability, and serialization."""

    def test_invalid_waveform_dimensions_rejected(self):
        """Input with temporal dimension != 360 must raise ValueError."""
        clf = MethodBResNetClassifier()
        rh = np.zeros((5, 4), dtype=np.float32)
        y = np.zeros(5, dtype=int)

        # 3D: 359 samples
        with pytest.raises(ValueError, match=r"Expected (temporal dimension 360|shape)"):
            clf.fit(np.zeros((5, 1, 359)), rh, y)

        # 2D: 361 samples
        with pytest.raises(ValueError, match=r"Expected (temporal dimension 360|shape)"):
            clf.fit(np.zeros((5, 361)), rh, y)

    def test_invalid_rhythm_dimensions_rejected(self):
        """Input with rhythm dimension != 4 must raise ValueError."""
        clf = MethodBResNetClassifier()
        wf = np.zeros((5, 1, 360), dtype=np.float32)
        y = np.zeros(5, dtype=int)

        # 3 features
        with pytest.raises(ValueError, match="Expected 4 rhythm features"):
            clf.fit(wf, np.zeros((5, 3)), y)

        # 5 features
        with pytest.raises(ValueError, match="Expected 4 rhythm features"):
            clf.fit(wf, np.zeros((5, 5)), y)

    def test_nan_and_inf_inputs_rejected(self):
        """NaN or Inf values in waveform or rhythm must raise ValueError."""
        clf = MethodBResNetClassifier()
        wf = np.zeros((5, 1, 360), dtype=np.float32)
        rh = np.zeros((5, 4), dtype=np.float32)
        y = np.zeros(5, dtype=int)

        wf_nan = wf.copy()
        wf_nan[0, 0, 100] = np.nan
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            clf.fit(wf_nan, rh, y)

        rh_inf = rh.copy()
        rh_inf[1, 2] = np.inf
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            clf.fit(wf, rh_inf, y)

    def test_invalid_labels_rejected(self):
        """Labels containing negative values, out of range (>4), or non-convertible must raise ValueError."""
        clf = MethodBResNetClassifier()
        wf = np.zeros((5, 1, 360), dtype=np.float32)
        rh = np.zeros((5, 4), dtype=np.float32)

        # Negative label
        with pytest.raises(ValueError, match="in range"):
            clf.fit(wf, rh, np.array([0, 1, -1, 2, 3]))

        # Label 5 (out of range [0, 4])
        with pytest.raises(ValueError, match="in range"):
            clf.fit(wf, rh, np.array([0, 1, 5, 2, 3]))

    def test_focal_loss_extreme_logits_numerical_stability(self):
        """Focal loss with huge logits (+100, -100) must compute finite scalar loss without NaN/Inf."""
        loss_fn = MultiClassFocalLoss(gamma=2.0)
        # Saturated logits
        logits_saturated = torch.tensor([
            [100.0, -100.0, -100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0, -100.0, -100.0],
            [-100.0, -100.0, 100.0, -100.0, -100.0],
        ], dtype=torch.float32)
        targets_matched = torch.tensor([0, 1, 2], dtype=torch.long)

        loss_matched = loss_fn(logits_saturated, targets_matched)
        assert torch.isfinite(loss_matched)
        assert loss_matched.item() < 1e-4, f"Loss should be ~0 for perfect confidence, got {loss_matched.item()}"

        # Inverted targets (worst-case penalty)
        targets_mismatched = torch.tensor([1, 2, 0], dtype=torch.long)
        loss_mismatched = loss_fn(logits_saturated, targets_mismatched)
        assert torch.isfinite(loss_mismatched)
        assert loss_mismatched.item() > 0.0

    def test_checkpoint_roundtrip_fidelity(self):
        """Verify model checkpoint saves and loads with bit-for-bit identical predictions."""
        rng = np.random.RandomState(42)
        n = 30
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=2, batch_size=32, random_state=42)
        clf.fit(X_wf, X_rh, y)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            clf.save_checkpoint(checkpoint_path)
            loaded_clf = MethodBResNetClassifier.load_checkpoint(checkpoint_path)

            assert loaded_clf.is_fitted
            np.testing.assert_array_equal(loaded_clf.fitted_classes_, clf.fitted_classes_)

            orig_proba = clf.predict_proba(X_wf, X_rh)
            loaded_proba = loaded_clf.predict_proba(X_wf, X_rh)
            np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)

            orig_preds = clf.predict(X_wf, X_rh)
            loaded_preds = loaded_clf.predict(X_wf, X_rh)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
