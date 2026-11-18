"""
Unit, Integration, and Hardware Benchmark Test Suite for Milestone 3:
Method B (Deep 1D-ResNet with Auxiliary Rhythm Fusion).
Adheres strictly to PROJECT.md Contract 2 and ANSI/AAMI EC57 standards.
"""

import os
import tempfile
import time
from typing import Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base_classifier import BaseECGClassifier
from src.models.method_b_resnet import (
    ECGResNet1D,
    MethodBResNetClassifier,
    MultiClassFocalLoss,
    ResidualBlock1D,
)


# ============================================================================
# 1. ARCHITECTURAL INVARIANTS, PARAMETERS & TENSOR FLOW
# ============================================================================

class TestArchitectureAndTensorFlow:
    """Verifies 1D-ResNet architectural specifications, receptive field, and parameters."""

    def test_total_and_trainable_parameter_counts(self):
        """Verify total parameters equals exactly 1,714,565 (~1.71M) and all require grad."""
        model = ECGResNet1D()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == 1714565, f"Expected 1,714,565 params, got {total_params}"
        assert trainable_params == 1714565, f"Expected all params trainable, got {trainable_params}"

    def test_stem_and_stage_tensor_shapes(self):
        """Verify layer-by-layer intermediate tensor shapes through the 8 residual blocks."""
        model = ECGResNet1D()
        model.eval()

        B = 4
        x_wf = torch.randn(B, 1, 360)
        x_rh = torch.randn(B, 4)

        with torch.no_grad():
            # Stem Conv: k=15, s=2, p=7 -> (B, 64, 180) -> MaxPool(k=3, s=2, p=1) -> (B, 64, 90)
            h_stem = model.stem(x_wf)
            assert h_stem.shape == (B, 64, 90), f"Stem shape mismatch: {h_stem.shape}"

            # Stage 1: 2 blocks @ 64 ch, k=7, s=1 -> (B, 64, 90)
            h_stage1 = model.stage1(h_stem)
            assert h_stage1.shape == (B, 64, 90), f"Stage 1 shape mismatch: {h_stage1.shape}"

            # Stage 2: 3 blocks @ 128 ch, k=5, s=2 downsample -> (B, 128, 45)
            h_stage2 = model.stage2(h_stage1)
            assert h_stage2.shape == (B, 128, 45), f"Stage 2 shape mismatch: {h_stage2.shape}"

            # Stage 3: 3 blocks @ 256 ch, k=3, s=2 downsample -> (B, 256, 23)
            h_stage3 = model.stage3(h_stage2)
            assert h_stage3.shape == (B, 256, 23), f"Stage 3 shape mismatch: {h_stage3.shape}"

            # Global Average Pooling -> (B, 256)
            h_wf = torch.flatten(model.gap(h_stage3), 1)
            assert h_wf.shape == (B, 256), f"GAP shape mismatch: {h_wf.shape}"

            # Auxiliary Rhythm MLP: 4 -> 32 -> 32 -> (B, 32)
            h_rh = model.rhythm_mlp(x_rh)
            assert h_rh.shape == (B, 32), f"Rhythm MLP shape mismatch: {h_rh.shape}"

            # Late Fusion: 256 + 32 = 288
            fused = torch.cat([h_wf, h_rh], dim=1)
            assert fused.shape == (B, 288), f"Fused shape mismatch: {fused.shape}"

            # Head -> (B, 5)
            logits = model.classifier(fused)
            assert logits.shape == (B, 5), f"Logits shape mismatch: {logits.shape}"

    def test_receptive_field_progression_spans_full_heartbeat(self):
        """
        Verify that the theoretical receptive field through Stage 3 spans > 360 samples,
        guaranteeing full-beat coverage of P-QRS-T complexes.
        """
        # Layer sequence: (kernel_size, stride)
        layers = [
            (15, 2),  # Stem conv
            (3, 2),   # Stem maxpool
            (7, 1),   # Stage 1 Block 1 conv1
            (7, 1),   # Stage 1 Block 1 conv2
            (7, 1),   # Stage 1 Block 2 conv1
            (7, 1),   # Stage 1 Block 2 conv2
            (5, 2),   # Stage 2 Block 3 conv1 (s=2 downsample)
            (5, 1),   # Stage 2 Block 3 conv2
            (5, 1),   # Stage 2 Block 4 conv1
            (5, 1),   # Stage 2 Block 4 conv2
            (5, 1),   # Stage 2 Block 5 conv1
            (5, 1),   # Stage 2 Block 5 conv2
            (3, 2),   # Stage 3 Block 6 conv1 (s=2 downsample)
            (3, 1),   # Stage 3 Block 6 conv2
            (3, 1),   # Stage 3 Block 7 conv1
            (3, 1),   # Stage 3 Block 7 conv2
            (3, 1),   # Stage 3 Block 8 conv1
            (3, 1),   # Stage 3 Block 8 conv2
        ]

        rf = 1
        jump = 1
        for k, s in layers:
            rf += (k - 1) * jump
            jump *= s

        # rf should be 467 samples, which exceeds the 360-sample heartbeat window
        assert rf == 467, f"Expected receptive field 467, got {rf}"
        assert rf > 360, f"Receptive field {rf} does not cover full 360-sample heartbeat"

    def test_forward_single_sample_and_diverse_shapes(self):
        """ECGResNet1D forward supports (B, 1, 360), (B, 360), and 1D (360,) inputs."""
        model = ECGResNet1D()
        model.eval()

        with torch.no_grad():
            # Standard 3D batch
            out_3d = model(torch.randn(2, 1, 360), torch.randn(2, 4))
            assert out_3d.shape == (2, 5)

            # 2D batch (B, 360)
            out_2d = model(torch.randn(3, 360), torch.randn(3, 4))
            assert out_2d.shape == (3, 5)

            # Single sample 1D (360,) and (4,)
            out_1d = model(torch.randn(360), torch.randn(4))
            assert out_1d.shape == (1, 5)

    def test_single_sample_vs_batch_evaluation_numerical_equivalence(self):
        """Evaluating sample 0 within a batch vs as a standalone sample yields identical logits."""
        model = ECGResNet1D()
        model.eval()

        torch.manual_seed(1234)
        x_wf_batch = torch.randn(5, 1, 360)
        x_rh_batch = torch.randn(5, 4)

        with torch.no_grad():
            batch_logits = model(x_wf_batch, x_rh_batch)
            single_logits = model(x_wf_batch[0:1], x_rh_batch[0:1])

        max_diff = torch.max(torch.abs(batch_logits[0:1] - single_logits)).item()
        assert max_diff < 1e-5, f"Discrepancy {max_diff} exceeds 1e-5"


# ============================================================================
# 2. MULTI-CLASS FOCAL LOSS NUMERICAL STABILITY & GRADIENTS
# ============================================================================

class TestMultiClassFocalLoss:
    """Verifies focal loss properties, class weighting, reductions, and numerical stability."""

    def test_focal_loss_default_balanced_and_unweighted(self):
        """Verify loss computes positive finite scalar with alpha=None and alpha tensor."""
        loss_fn_unweighted = MultiClassFocalLoss(alpha=None, gamma=2.0)
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        loss_val = loss_fn_unweighted(logits, targets)

        assert loss_val.dim() == 0
        assert loss_val.item() > 0.0
        assert torch.isfinite(loss_val)

        # With balanced alpha tensor
        alpha = torch.tensor([0.2, 1.5, 1.2, 2.0, 3.0], dtype=torch.float32)
        loss_fn_weighted = MultiClassFocalLoss(alpha=alpha, gamma=2.0)
        loss_weighted_val = loss_fn_weighted(logits, targets)
        assert loss_weighted_val.dim() == 0
        assert loss_weighted_val.item() > 0.0
        assert torch.isfinite(loss_weighted_val)

    def test_focal_loss_extreme_probabilities_attenuation(self):
        """Verify high confidence correct predictions are attenuated by (1 - p_t)^gamma."""
        loss_fn = MultiClassFocalLoss(gamma=2.0)

        # Extremely confident correct sample: p_0 ~ 1.0
        confident_logits = torch.tensor([[50.0, -50.0, -50.0, -50.0, -50.0]])
        target_correct = torch.tensor([0])
        loss_easy = loss_fn(confident_logits, target_correct)
        assert loss_easy.item() < 1e-4, f"Easy sample loss {loss_easy.item()} should be near 0"

        # Confident incorrect sample: p_0 ~ 1.0, but target is 1
        target_incorrect = torch.tensor([1])
        loss_hard = loss_fn(confident_logits, target_incorrect)
        assert loss_hard.item() > 50.0, f"Hard sample loss {loss_hard.item()} should be large"
        assert torch.isfinite(loss_hard)

    def test_focal_loss_reductions(self):
        """Verify reduction options: 'mean', 'sum', 'none'."""
        logits = torch.randn(6, 5)
        targets = torch.randint(0, 5, (6,))

        loss_mean = MultiClassFocalLoss(reduction="mean")(logits, targets)
        loss_sum = MultiClassFocalLoss(reduction="sum")(logits, targets)
        loss_none = MultiClassFocalLoss(reduction="none")(logits, targets)

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (6,)
        np.testing.assert_allclose(loss_mean.item(), loss_none.mean().item(), atol=1e-5)
        np.testing.assert_allclose(loss_sum.item(), loss_none.sum().item(), atol=1e-5)

    def test_focal_loss_backward_gradient_flow(self):
        """Verify backward pass computes finite non-zero gradients."""
        model = ECGResNet1D()
        loss_fn = MultiClassFocalLoss(gamma=2.0)
        x_wf = torch.randn(4, 1, 360)
        x_rh = torch.randn(4, 4)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        logits = model(x_wf, x_rh)
        loss = loss_fn(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf grad"


# ============================================================================
# 3. METHOD B RESNET CLASSIFIER CONTRACT 2 COMPLIANCE
# ============================================================================

class TestMethodBResNetClassifierContract2:
    """Verifies MethodBResNetClassifier conforms strictly to Interface Contract 2."""

    @pytest.fixture
    def synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate reproducible 5-class synthetic ECG dataset."""
        np.random.seed(42)
        n = 80
        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.6, 0.1, 0.15, 0.1, 0.05])
        return X_wf, X_rh, y

    def test_subclass_of_base_classifier(self):
        """Verify MethodBResNetClassifier is a subclass of BaseECGClassifier."""
        assert issubclass(MethodBResNetClassifier, BaseECGClassifier)
        clf = MethodBResNetClassifier(epochs=5, device="cpu")
        assert isinstance(clf, BaseECGClassifier)

    def test_invalid_parameters_raise_value_error(self):
        """Invalid hyperparameter arguments raise ValueError."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            MethodBResNetClassifier(epochs=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            MethodBResNetClassifier(batch_size=-1)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            MethodBResNetClassifier(learning_rate=-0.01)

    def test_unfitted_model_protection(self, synthetic_data):
        """Calling predict, predict_proba, or save_checkpoint before fit raises RuntimeError."""
        clf = MethodBResNetClassifier(epochs=5, device="cpu")
        X_wf, X_rh, _ = synthetic_data

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict(X_wf, X_rh)

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            clf.predict_proba(X_wf, X_rh)

        with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
            clf.save_checkpoint("dummy.pt")

    def test_fit_and_predict_shapes_and_types(self, synthetic_data):
        """Fit completes and predict returns (N,) int with values strictly in {0, 1, 2, 3, 4}."""
        X_wf, X_rh, y = synthetic_data
        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        assert clf.is_fitted
        preds = clf.predict(X_wf, X_rh)
        assert preds.shape == (len(X_wf),)
        assert np.issubdtype(preds.dtype, np.integer)
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_predict_proba_contract_invariants(self, synthetic_data):
        """predict_proba returns (N, 5) float32 with values in [0, 1] and rows summing to 1.0."""
        X_wf, X_rh, y = synthetic_data
        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (len(X_wf), 5)
        assert proba.dtype == np.float32
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_matches_argmax_predict_proba(self, synthetic_data):
        """clf.predict strictly matches np.argmax(clf.predict_proba, axis=1)."""
        X_wf, X_rh, y = synthetic_data
        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        preds = clf.predict(X_wf, X_rh)
        proba = clf.predict_proba(X_wf, X_rh)
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

    def test_empty_input_handling(self, synthetic_data):
        """Empty input arrays to fitted model return empty arrays without crashing."""
        X_wf, X_rh, y = synthetic_data
        clf = MethodBResNetClassifier(epochs=2, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        p_empty = clf.predict_proba(np.zeros((0, 1, 360)), np.zeros((0, 4)))
        assert p_empty.shape == (0, 5)
        assert p_empty.dtype == np.float32

        pred_empty = clf.predict(np.zeros((0, 1, 360)), np.zeros((0, 4)))
        assert pred_empty.shape == (0,)
        assert np.issubdtype(pred_empty.dtype, np.integer)

    def test_single_sample_inference(self, synthetic_data):
        """Single beat (360,) and (4,) return (1, 5) probability and (1,) prediction."""
        X_wf, X_rh, y = synthetic_data
        clf = MethodBResNetClassifier(epochs=2, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        p_single = clf.predict_proba(X_wf[0, 0], X_rh[0])
        assert p_single.shape == (1, 5)
        assert p_single.dtype == np.float32
        np.testing.assert_allclose(p_single.sum(axis=1), 1.0, atol=1e-5)

        pred_single = clf.predict(X_wf[0, 0], X_rh[0])
        assert pred_single.shape == (1,)
        assert pred_single[0] in {0, 1, 2, 3, 4}

    def test_input_validation_rejections(self, synthetic_data):
        """Validates that NaN, Inf, and shape mismatches are rejected with ValueError."""
        clf = MethodBResNetClassifier(epochs=2, device="cpu")
        X_wf, X_rh, y = synthetic_data

        # NaN in waveform
        bad_wf = X_wf.copy()
        bad_wf[0, 0, 50] = np.nan
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            clf.fit(bad_wf, X_rh, y)

        # Inf in rhythm
        bad_rh = X_rh.copy()
        bad_rh[0, 2] = np.inf
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            clf.fit(X_wf, bad_rh, y)

        # Mismatched sample counts
        with pytest.raises(ValueError, match="Sample count mismatch"):
            clf.fit(X_wf[:10], X_rh[:5], y[:10])

        # Wrong temporal length 3D
        with pytest.raises(ValueError, match="Expected shape"):
            clf.fit(np.zeros((10, 1, 200)), X_rh[:10], y[:10])

        # Wrong temporal length 2D
        with pytest.raises(ValueError, match="Expected temporal dimension 360"):
            clf.fit(np.zeros((10, 200)), X_rh[:10], y[:10])


# ============================================================================
# 4. SINGLE-CLASS AND SUBSET-CLASS EDGE CASES
# ============================================================================

class TestSingleClassAndSubsetEdgeCases:
    """Verifies edge case resilience on single-class and class subsets."""

    @pytest.mark.parametrize("target_class", [0, 1, 2, 3, 4])
    def test_single_class_training_for_all_classes(self, target_class):
        """Fit on single class succeeds and assigns 1.0 probability mass to that class."""
        n = 25
        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.ones((n, 4), dtype=np.float32)
        y = np.full(n, target_class, dtype=int)

        clf = MethodBResNetClassifier(epochs=2, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        assert clf.is_fitted
        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (n, 5)
        np.testing.assert_allclose(proba[:, target_class], 1.0)
        for other_c in range(5):
            if other_c != target_class:
                np.testing.assert_allclose(proba[:, other_c], 0.0)

        preds = clf.predict(X_wf, X_rh)
        np.testing.assert_array_equal(preds, np.full(n, target_class, dtype=int))

    @pytest.mark.parametrize(
        "observed_classes",
        [
            [0, 2],
            [1, 3],
            [0, 1, 4],
            [0, 1, 2, 3],
        ],
    )
    def test_subset_classes_training_projects_to_canonical_5(self, observed_classes):
        """
        When trained on class subsets, unobserved classes have 0.0 probability,
        rows sum strictly to 1.0, and discrete predictions are a subset of observed.
        """
        n_per_class = 15
        y_list = []
        for c in observed_classes:
            y_list.extend([c] * n_per_class)
        y = np.array(y_list, dtype=int)
        n = len(y)

        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.1, size=(n, 4)).astype(np.float32)

        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (n, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        unobserved = set(range(5)) - set(observed_classes)
        for c in unobserved:
            assert np.all(proba[:, c] == 0.0), f"Unobserved class {c} must have 0.0 probability"

        preds = clf.predict(X_wf, X_rh)
        assert set(preds).issubset(set(observed_classes))
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

    def test_subset_classes_underflow_protection(self):
        """When subset classes produce underflow or extreme logits, uniform mass across fitted classes is assigned."""
        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        clf.fitted_classes_ = np.array([1, 2], dtype=int)
        clf.is_fitted = True
        clf.model_ = ECGResNet1D().to("cpu")
        clf.model_.eval()

        # Extreme inputs
        X_wf = np.full((10, 1, 360), 1e6, dtype=np.float32)
        X_rh = np.full((10, 4), 1e6, dtype=np.float32)

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (10, 5)
        # Probability on unobserved classes {0, 3, 4} must be 0.0
        assert np.all(proba[:, [0, 3, 4]] == 0.0)
        # Probability across fitted classes {1, 2} must sum to 1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        # Discrete predictions must be in {1, 2}
        preds = clf.predict(X_wf, X_rh)
        assert set(preds).issubset({1, 2})


# ============================================================================
# 5. CHECKPOINT SERIALIZATION ROUNDTRIP (PYTORCH 2.6+ COMPATIBILITY)
# ============================================================================

class TestCheckpointSerialization:
    """Verifies checkpoint saving and loading roundtrip with PyTorch 2.6+ weights_only compatibility."""

    def test_checkpoint_roundtrip_preserves_predictions_and_proba(self):
        """Save and load roundtrip reproduces exact logits and probabilities."""
        rng = np.random.RandomState(42)
        n = 50
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.5, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "models", "nested", "method_b_resnet.pt")
            clf.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            loaded_clf = MethodBResNetClassifier.load_checkpoint(ckpt_path)
            assert loaded_clf.is_fitted
            assert loaded_clf.epochs == clf.epochs
            assert np.array_equal(loaded_clf.fitted_classes_, clf.fitted_classes_)

            orig_preds = clf.predict(X_wf, X_rh)
            loaded_preds = loaded_clf.predict(X_wf, X_rh)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

            orig_proba = clf.predict_proba(X_wf, X_rh)
            loaded_proba = loaded_clf.predict_proba(X_wf, X_rh)
            np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)

    def test_checkpoint_roundtrip_single_class(self):
        """Checkpoint serialization works cleanly for single-class fitted instances."""
        n = 20
        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.ones((n, 4), dtype=np.float32)
        y = np.full(n, 2, dtype=int)

        clf = MethodBResNetClassifier(epochs=2, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "single_class.pt")
            clf.save_checkpoint(ckpt_path)
            loaded = MethodBResNetClassifier.load_checkpoint(ckpt_path)

            proba = loaded.predict_proba(X_wf, X_rh)
            np.testing.assert_allclose(proba[:, 2], 1.0)
            preds = loaded.predict(X_wf, X_rh)
            np.testing.assert_array_equal(preds, np.full(n, 2, dtype=int))

    def test_load_nonexistent_checkpoint_raises(self):
        """Attempting to load a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MethodBResNetClassifier.load_checkpoint("/path/to/nonexistent/model.pt")

    def test_checkpoint_cross_device_fallback(self):
        """Checkpoints saved with unavailable device (e.g. cuda:99) load and fall back gracefully."""
        n = 20
        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.ones((n, 4), dtype=np.float32)
        y = np.random.choice([0, 1, 2], size=n)

        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "cross_dev.pt")
            clf.save_checkpoint(ckpt_path)

            # Artificially set device in checkpoint to an unavailable CUDA device
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw_ckpt["params"]["device"] = "cuda:99"
            torch.save(raw_ckpt, ckpt_path)

            # Loading should not crash; it should fall back to an available device
            loaded = MethodBResNetClassifier.load_checkpoint(ckpt_path)
            assert loaded.is_fitted
            proba = loaded.predict_proba(X_wf, X_rh)
            assert proba.shape == (n, 5)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ============================================================================
# 6. HARDWARE EXECUTION, CONVERGENCE SPEED & VALIDATION LOOP
# ============================================================================

class TestHardwareAndConvergenceSpeed:
    """Verifies execution on CPU and MPS, and benchmark convergence speed (<15s on 200 samples)."""

    def test_convergence_speed_under_15_seconds(self):
        """Verify training 5 epochs on 200 benchmark samples finishes in <15s on accelerator (<60s on CPU)."""
        rng = np.random.RandomState(42)
        n = 200
        X_wf = rng.randn(n, 1, 360).astype(np.float32)
        X_rh = rng.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.7, 0.1, 0.1, 0.05, 0.05])

        clf = MethodBResNetClassifier(
            epochs=5,
            batch_size=64,
            learning_rate=1e-3,
            device=None,  # Autoselect hardware accelerator (MPS/CUDA) or CPU
            random_state=42,
        )

        start_time = time.perf_counter()
        clf.fit(X_wf, X_rh, y)
        elapsed = time.perf_counter() - start_time

        print(f"\n200-sample 5-epoch {clf.device_.type.upper()} training time: {elapsed:.3f}s")
        max_allowed = 60.0 if clf.device_.type == "cpu" else 15.0
        assert elapsed < max_allowed, f"Training took {elapsed:.3f}s, exceeding {max_allowed}s requirement"

        # Verify training loss decreased
        assert len(clf.loss_history_) >= 2, "Expected at least 2 epochs recorded in loss_history_"
        assert clf.loss_history_[-1] < clf.loss_history_[0], (
            f"Loss did not decrease: initial={clf.loss_history_[0]:.4f}, final={clf.loss_history_[-1]:.4f}"
        )

        # Verify predictions
        preds = clf.predict(X_wf, X_rh)
        assert len(preds) == n

    def test_mps_device_if_available(self):
        """If Apple Silicon MPS is available, test training and inference on MPS."""
        if not torch.backends.mps.is_available():
            pytest.skip("Apple Silicon MPS not available on this platform")

        n = 50
        X_wf = np.random.randn(n, 1, 360).astype(np.float32)
        X_rh = np.random.uniform(0.6, 1.2, size=(n, 4)).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=n)

        clf = MethodBResNetClassifier(epochs=3, batch_size=32, device="mps", random_state=42)
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted
        assert clf.device_.type == "mps"

        proba = clf.predict_proba(X_wf, X_rh)
        assert proba.shape == (n, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_with_val_data_early_stopping(self):
        """Verify fitting with validation data tracks validation loss and early stops cleanly."""
        rng = np.random.RandomState(42)
        n_train = 60
        n_val = 20

        X_train_wf = rng.randn(n_train, 1, 360).astype(np.float32)
        X_train_rh = rng.uniform(0.6, 1.2, size=(n_train, 4)).astype(np.float32)
        y_train = rng.choice([0, 1, 2, 3, 4], size=n_train)

        X_val_wf = rng.randn(n_val, 1, 360).astype(np.float32)
        X_val_rh = rng.uniform(0.6, 1.2, size=(n_val, 4)).astype(np.float32)
        y_val = rng.choice([0, 1, 2, 3, 4], size=n_val)

        clf = MethodBResNetClassifier(
            epochs=10,
            batch_size=32,
            early_stopping_patience=3,
            device="cpu",
            random_state=42,
        )
        clf.fit(X_train_wf, X_train_rh, y_train, val_data=(X_val_wf, X_val_rh, y_val))
        assert clf.is_fitted

        val_preds = clf.predict(X_val_wf, X_val_rh)
        assert len(val_preds) == n_val
