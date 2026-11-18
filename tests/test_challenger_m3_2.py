"""
Empirical Challenger Test Suite for Milestone 3 (Method B: Deep 1D-ResNet with Rhythm Fusion).
Tests:
1. Imbalance & Focal Loss Robustness under severe de Chazal distribution (90% N, 2% S, 6% V, 1% F, 1% Q).
2. Serialization & Hardware Portability (MPS/CPU roundtrip, <1e-6 error, PyTorch 2.6+ unpickler compatibility).
3. Determinism & Reproducibility (seed invariance, bit-for-bit prediction parity, eval mode freeze).
"""

import os
import tempfile
from typing import Tuple
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
# HELPER: SYNTHETIC DATA GENERATORS FOR ADVERSARIAL TESTING
# ============================================================================

def generate_de_chazal_imbalanced_dataset(
    n_samples: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates de Chazal inter-patient class imbalance:
    Class 0 (Normal): 90%
    Class 1 (Supraventricular ectopic S): 2%
    Class 2 (Ventricular ectopic V): 6%
    Class 3 (Fusion F): 1%
    Class 4 (Unknown Q): 1%
    """
    rng = np.random.RandomState(random_state)
    # Approximate counts matching de Chazal distribution
    # For n=200: 180, 4, 12, 2, 2
    c0 = int(n_samples * 0.90)
    c1 = max(1, int(n_samples * 0.02))
    c2 = max(1, int(n_samples * 0.06))
    c3 = max(1, int(n_samples * 0.01))
    c4 = max(1, n_samples - (c0 + c1 + c2 + c3))
    counts = [c0, c1, c2, c3, c4]

    X_wf_list = []
    X_rh_list = []
    y_list = []

    t = np.linspace(0, 1, 360)

    for c, count in enumerate(counts):
        for _ in range(count):
            if c == 0:  # Normal sinus
                sig = np.sin(2 * np.pi * 5 * t)
                rh = [1.0, 1.0, 0.0, 1.0]
            elif c == 1:  # S: Premature timing, normal morphology
                sig = np.sin(2 * np.pi * 5 * t)
                rh = [0.6, 1.2, -0.4, 0.5]
            elif c == 2:  # V: Wide QRS, compensatory pause
                sig = 2.0 * np.sin(2 * np.pi * 2 * t)
                rh = [1.0, 1.5, 0.5, 1.5]
            elif c == 3:  # F: Intermediate fusion morphology
                sig = 1.5 * np.sin(2 * np.pi * 3.5 * t)
                rh = [0.8, 1.3, 0.1, 0.8]
            else:  # Q: Anomalous / noisy
                sig = rng.randn(360) * 0.5
                rh = [1.0, 1.0, 0.0, 1.0]

            sig += rng.randn(360) * 0.05
            # Z-score normalize as required by Contract 1
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            X_wf_list.append(sig.reshape(1, 360))
            X_rh_list.append(rh)
            y_list.append(c)

    # Shuffle
    indices = rng.permutation(len(y_list))
    X_wf = np.array(X_wf_list, dtype=np.float32)[indices]
    X_rh = np.array(X_rh_list, dtype=np.float32)[indices]
    y = np.array(y_list, dtype=int)[indices]
    return X_wf, X_rh, y


def generate_balanced_test_dataset(
    samples_per_class: int = 10,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a balanced test set across all 5 AAMI classes."""
    rng = np.random.RandomState(random_state)
    X_wf_list = []
    X_rh_list = []
    y_list = []
    t = np.linspace(0, 1, 360)

    for c in range(5):
        for _ in range(samples_per_class):
            if c == 0:
                sig = np.sin(2 * np.pi * 5 * t)
                rh = [1.0, 1.0, 0.0, 1.0]
            elif c == 1:
                sig = np.sin(2 * np.pi * 5 * t)
                rh = [0.6, 1.2, -0.4, 0.5]
            elif c == 2:
                sig = 2.0 * np.sin(2 * np.pi * 2 * t)
                rh = [1.0, 1.5, 0.5, 1.5]
            elif c == 3:
                sig = 1.5 * np.sin(2 * np.pi * 3.5 * t)
                rh = [0.8, 1.3, 0.1, 0.8]
            else:
                sig = rng.randn(360) * 0.5
                rh = [1.0, 1.0, 0.0, 1.0]

            sig += rng.randn(360) * 0.05
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            X_wf_list.append(sig.reshape(1, 360))
            X_rh_list.append(rh)
            y_list.append(c)

    return (
        np.array(X_wf_list, dtype=np.float32),
        np.array(X_rh_list, dtype=np.float32),
        np.array(y_list, dtype=int),
    )


# ============================================================================
# 1. IMBALANCE & FOCAL LOSS ROBUSTNESS TESTS
# ============================================================================

class TestImbalanceAndFocalLossRobustness:
    """Empirically tests Focal Loss behaviour under severe de Chazal class imbalance."""

    def test_focal_loss_gradient_scaling_under_severe_imbalance(self):
        """
        Verify that Focal Loss down-weights gradients from easy/well-classified majority samples
        and maintains strong gradient signal for hard minority samples.
        """
        gamma = 2.0
        alpha = torch.tensor([0.2, 2.0, 1.0, 2.0, 2.0], dtype=torch.float32)
        focal_loss_fn = MultiClassFocalLoss(alpha=alpha, gamma=gamma, reduction="none")

        # Simulate batch of 2 samples:
        # Sample 0: Majority class 0, well classified (high probability p_t = 0.95)
        # Sample 1: Minority class 1, misclassified (low probability p_t = 0.10)
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0],   # Class 0 dominant
                [4.0, 0.0, 0.0, 0.0, 0.0],   # Class 0 dominant, but target is Class 1!
            ],
            requires_grad=True,
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss = focal_loss_fn(logits, targets)
        loss.backward(torch.ones_like(loss))

        grad_sample_0 = logits.grad[0].norm().item()
        grad_sample_1 = logits.grad[1].norm().item()

        # Gradient on the hard minority sample must be significantly higher than on easy majority sample
        assert grad_sample_1 > grad_sample_0 * 5.0, (
            f"Expected minority gradient >> majority gradient, got {grad_sample_1:.4f} vs {grad_sample_0:.4f}"
        )

    def test_model_avoids_majority_class_collapse_under_de_chazal_imbalance(self):
        """
        Verify that MethodBResNetClassifier trained on 90% Class 0 avoids collapsing
        to predicting 100% Class 0, and successfully detects minority classes on a test set.
        """
        X_train_wf, X_train_rh, y_train = generate_de_chazal_imbalanced_dataset(
            n_samples=200, random_state=42
        )
        X_test_wf, X_test_rh, y_test = generate_balanced_test_dataset(
            samples_per_class=10, random_state=123
        )

        clf = MethodBResNetClassifier(
            epochs=5,
            batch_size=32,
            learning_rate=2e-3,
            gamma=2.0,
            alpha="balanced",
            random_state=42,
        )
        clf.fit(X_train_wf, X_train_rh, y_train)

        test_preds = clf.predict(X_test_wf, X_test_rh)
        unique_preds, pred_counts = np.unique(test_preds, return_counts=True)

        # 1. Assert no total collapse to Class 0
        assert len(unique_preds) > 1, f"Collapsed to single class: {unique_preds}"
        # 2. Assert minority classes (classes 1, 2, 3, or 4) are actively predicted
        minority_predicted = any(c in unique_preds for c in [1, 2, 3, 4])
        assert minority_predicted, f"No minority classes predicted: {unique_preds}"
        # 3. Macro accuracy must exceed trivial majority guess (0.2 on 5-class balanced test set)
        macro_acc = np.mean(test_preds == y_test)
        assert macro_acc > 0.40, f"Expected macro accuracy > 0.40 under focal loss, got {macro_acc:.3f}"

    def test_focal_loss_numerical_extremes_and_boundary_conditions(self):
        """
        Stress test MultiClassFocalLoss with extreme logits (+100.0, -100.0)
        to verify that log_softmax and clamp prevent NaN or Inf blowups.
        """
        focal_fn = MultiClassFocalLoss(gamma=2.0, reduction="mean")

        # Extremely confident correct logits -> p_t ~ 1.0 -> loss ~ 0
        extreme_correct = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0]], requires_grad=True)
        loss_correct = focal_fn(extreme_correct, torch.tensor([0]))
        loss_correct.backward()

        assert not torch.isnan(loss_correct), "NaN in focal loss for extreme correct logits"
        assert not torch.isinf(loss_correct), "Inf in focal loss for extreme correct logits"
        assert loss_correct.item() >= 0.0, "Loss must be non-negative"
        assert loss_correct.item() < 1e-4, f"Expected loss near zero, got {loss_correct.item()}"
        assert not torch.isnan(extreme_correct.grad).any(), "NaN in gradients"

        # Extremely confident incorrect logits -> p_t ~ 0.0 -> loss should be finite
        extreme_wrong = torch.tensor([[-100.0, 100.0, -100.0, -100.0, -100.0]], requires_grad=True)
        loss_wrong = focal_fn(extreme_wrong, torch.tensor([0]))
        loss_wrong.backward()

        assert not torch.isnan(loss_wrong), "NaN in focal loss for extreme incorrect logits"
        assert not torch.isinf(loss_wrong), "Inf in focal loss for extreme incorrect logits"
        assert loss_wrong.item() > 0.0, "Loss must be positive"
        assert not torch.isnan(extreme_wrong.grad).any(), "NaN in gradients"

    def test_alpha_computation_with_zero_count_classes(self):
        """
        Verify that _compute_alpha_weights gracefully handles absent classes
        (count=0) without division-by-zero or NaN weights.
        """
        clf = MethodBResNetClassifier(alpha="balanced")
        # Only classes 0 and 2 are present; 1, 3, 4 have count 0
        y = np.array([0, 0, 0, 2, 2, 0, 0, 2], dtype=int)
        alpha = clf._compute_alpha_weights(y)

        assert alpha is not None
        assert alpha.shape == (5,), f"Expected shape (5,), got {alpha.shape}"
        assert not torch.isnan(alpha).any(), "Alpha contains NaNs"
        assert not torch.isinf(alpha).any(), "Alpha contains Infs"
        assert (alpha > 0).all(), "All alpha weights must be strictly positive"


# ============================================================================
# 2. SERIALIZATION & HARDWARE PORTABILITY TESTS
# ============================================================================

class TestSerializationAndHardwarePortability:
    """Empirically tests serialization roundtrips, CPU/MPS parity, and PyTorch 2.6 unpickler."""

    def test_checkpoint_roundtrip_cpu_logits_parity(self):
        """
        Save model checkpoint on CPU, load on CPU, and verify logits match to < 1e-6 error.
        """
        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        X_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_rh = np.random.randn(20, 4).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=20)
        clf.fit(X_wf, X_rh, y)

        X_eval_wf = np.random.randn(30, 1, 360).astype(np.float32)
        X_eval_rh = np.random.randn(30, 4).astype(np.float32)

        clf.model_.eval()
        with torch.no_grad():
            orig_logits = clf.model_(torch.from_numpy(X_eval_wf), torch.from_numpy(X_eval_rh)).numpy()
        orig_proba = clf.predict_proba(X_eval_wf, X_eval_rh)
        orig_preds = clf.predict(X_eval_wf, X_eval_rh)

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            clf.save_checkpoint(tmp.name)

            loaded = MethodBResNetClassifier.load_checkpoint(tmp.name)
            assert loaded.is_fitted is True
            np.testing.assert_array_equal(loaded.fitted_classes_, clf.fitted_classes_)

            loaded.model_.eval()
            with torch.no_grad():
                loaded_logits = loaded.model_(
                    torch.from_numpy(X_eval_wf), torch.from_numpy(X_eval_rh)
                ).numpy()
            loaded_proba = loaded.predict_proba(X_eval_wf, X_eval_rh)
            loaded_preds = loaded.predict(X_eval_wf, X_eval_rh)

        max_logit_error = np.max(np.abs(orig_logits - loaded_logits))
        max_proba_error = np.max(np.abs(orig_proba - loaded_proba))

        assert max_logit_error < 1e-6, f"Logit parity error {max_logit_error} >= 1e-6"
        assert max_proba_error < 1e-6, f"Proba parity error {max_proba_error} >= 1e-6"
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_checkpoint_mps_to_cpu_portability(self):
        """
        Save model checkpoint trained on Apple Silicon MPS, load on CPU,
        and verify logits match to < 1e-5 error across hardware boundaries.
        """
        if not torch.backends.mps.is_available():
            pytest.skip("Apple Silicon MPS not available on this host")

        clf_mps = MethodBResNetClassifier(epochs=1, device="mps", random_state=42)
        X_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_rh = np.random.randn(20, 4).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=20)
        clf_mps.fit(X_wf, X_rh, y)

        X_eval_wf = np.random.randn(25, 1, 360).astype(np.float32)
        X_eval_rh = np.random.randn(25, 4).astype(np.float32)

        clf_mps.model_.eval()
        with torch.no_grad():
            mps_logits = clf_mps.model_(
                torch.from_numpy(X_eval_wf).to("mps"),
                torch.from_numpy(X_eval_rh).to("mps"),
            ).cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            clf_mps.save_checkpoint(tmp.name)

            # Explicitly force CPU loading
            loaded_cpu = MethodBResNetClassifier.load_checkpoint(tmp.name)
            loaded_cpu.device_ = torch.device("cpu")
            loaded_cpu.model_.to("cpu")
            loaded_cpu.model_.eval()

            with torch.no_grad():
                cpu_logits = loaded_cpu.model_(
                    torch.from_numpy(X_eval_wf), torch.from_numpy(X_eval_rh)
                ).numpy()

        diff = np.max(np.abs(mps_logits - cpu_logits))
        assert diff < 1e-5, f"Cross-hardware MPS->CPU error {diff} exceeded 1e-5 tolerance"

    def test_pytorch_2_6_unpickler_security_weights_only(self):
        """
        Verify checkpoint serialization conforms to PyTorch 2.6+ unpickler security policy
        (both weights_only=True and weights_only=False load cleanly).
        """
        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        clf.fit(
            np.random.randn(10, 1, 360).astype(np.float32),
            np.random.randn(10, 4).astype(np.float32),
            np.random.randint(0, 5, 10),
        )

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            clf.save_checkpoint(tmp.name)

            # Test PyTorch 2.6+ weights_only=True compatibility
            raw_weights_only = torch.load(tmp.name, weights_only=True)
            assert isinstance(raw_weights_only, dict)
            assert "model_state_dict" in raw_weights_only

            # Test MethodBResNetClassifier load_checkpoint
            loaded = MethodBResNetClassifier.load_checkpoint(tmp.name)
            assert loaded.is_fitted is True

    def test_corrupted_checkpoint_error_handling(self):
        """Verify proper error handling for non-existent and corrupt checkpoints."""
        with pytest.raises(FileNotFoundError):
            MethodBResNetClassifier.load_checkpoint("/nonexistent/path/model.pt")

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp.write(b"NOT_A_VALID_TORCH_CHECKPOINT")
            tmp.flush()
            with pytest.raises(Exception):
                MethodBResNetClassifier.load_checkpoint(tmp.name)


# ============================================================================
# 3. DETERMINISM & REPRODUCIBILITY TESTS
# ============================================================================

class TestDeterminismAndReproducibility:
    """Empirically tests reproducible training seeds and evaluation determinism."""

    def test_identical_random_seeds_produce_identical_weights_and_predictions(self):
        """
        Verify that two models trained on identical data with identical random seeds
        produce bit-for-bit identical weights and predictions.
        """
        X_wf = np.random.randn(30, 1, 360).astype(np.float32)
        X_rh = np.random.randn(30, 4).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=30)

        clf1 = MethodBResNetClassifier(epochs=2, batch_size=16, random_state=42, device="cpu")
        clf1.fit(X_wf, X_rh, y)

        clf2 = MethodBResNetClassifier(epochs=2, batch_size=16, random_state=42, device="cpu")
        clf2.fit(X_wf, X_rh, y)

        # Check weight tensors
        for (k1, v1), (k2, v2) in zip(
            clf1.model_.state_dict().items(), clf2.model_.state_dict().items()
        ):
            assert torch.equal(v1, v2), f"Weight mismatch in layer {k1} for identical seeds"

        # Check prediction probabilities
        p1 = clf1.predict_proba(X_wf, X_rh)
        p2 = clf2.predict_proba(X_wf, X_rh)
        assert np.max(np.abs(p1 - p2)) == 0.0, "Predict proba non-identical for identical seeds"

    def test_different_random_seeds_produce_different_initializations(self):
        """Verify that varying the random seed produces distinct model weights."""
        X_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_rh = np.random.randn(20, 4).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3, 4], size=20)

        clf1 = MethodBResNetClassifier(epochs=2, batch_size=10, random_state=42, device="cpu")
        clf1.fit(X_wf, X_rh, y)

        clf3 = MethodBResNetClassifier(epochs=2, batch_size=10, random_state=999, device="cpu")
        clf3.fit(X_wf, X_rh, y)

        weights_differ = any(
            not torch.allclose(v1.float(), v3.float(), atol=1e-4)
            for (k1, v1), (k3, v3) in zip(
                clf1.model_.state_dict().items(), clf3.model_.state_dict().items()
            )
        )
        assert weights_differ, "Different seeds unexpectedly produced identical weights"

    def test_eval_mode_inference_frozen_determinism(self):
        """
        Verify that multiple forward passes in eval mode produce 0 variance
        (proving dropout is deactivated and BatchNorm running stats are frozen).
        """
        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        clf.fit(
            np.random.randn(15, 1, 360).astype(np.float32),
            np.random.randn(15, 4).astype(np.float32),
            np.random.randint(0, 5, 15),
        )

        X_eval_wf = np.random.randn(20, 1, 360).astype(np.float32)
        X_eval_rh = np.random.randn(20, 4).astype(np.float32)

        p1 = clf.predict_proba(X_eval_wf, X_eval_rh)
        p2 = clf.predict_proba(X_eval_wf, X_eval_rh)
        p3 = clf.predict_proba(X_eval_wf, X_eval_rh)

        assert np.max(np.abs(p1 - p2)) == 0.0, "Stochastic variance detected between p1 and p2"
        assert np.max(np.abs(p2 - p3)) == 0.0, "Stochastic variance detected between p2 and p3"

    def test_batch_size_invariance_in_eval_mode(self):
        """
        Verify that evaluating samples in a single batch vs sample-by-sample
        yields identical probabilities (< 1e-6 max difference).
        """
        clf = MethodBResNetClassifier(epochs=1, device="cpu", random_state=42)
        clf.fit(
            np.random.randn(15, 1, 360).astype(np.float32),
            np.random.randn(15, 4).astype(np.float32),
            np.random.randint(0, 5, 15),
        )

        n_eval = 15
        X_eval_wf = np.random.randn(n_eval, 1, 360).astype(np.float32)
        X_eval_rh = np.random.randn(n_eval, 4).astype(np.float32)

        batched_proba = clf.predict_proba(X_eval_wf, X_eval_rh)

        single_probas = []
        for i in range(n_eval):
            p_single = clf.predict_proba(X_eval_wf[i : i + 1], X_eval_rh[i : i + 1])
            single_probas.append(p_single[0])
        single_probas = np.array(single_probas)

        max_diff = np.max(np.abs(batched_proba - single_probas))
        assert max_diff < 1e-6, f"Batch size invariance violated: max_diff = {max_diff}"
