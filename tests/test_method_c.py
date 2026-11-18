"""
Unit and Integration Tests for Method C: Hybrid 1D-CNN + BiLSTM + Bahdanau Attention.
Verifies tensor shapes, temporal attention normalization, explainability upsampling,
Contract 2 compliance, and checkpoint serialization.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.models.method_c_hybrid_attn import (
    BahdanauTemporalAttention,
    CNNBiLSTMAttention1D,
    MethodCHybridAttentionClassifier,
)


def _generate_synthetic_dataset(n_samples: int = 80, classes=(0, 1, 2, 3, 4), seed: int = 42):
    rng = np.random.default_rng(seed)
    X_wf = rng.standard_normal((n_samples, 1, 360)).astype(np.float32)
    X_rh = rng.uniform(0.5, 1.2, size=(n_samples, 4)).astype(np.float32)
    y = rng.choice(classes, size=n_samples).astype(int)
    # Inject synthetic class signatures
    for i, label in enumerate(y):
        t = np.linspace(0, 2 * np.pi, 360, dtype=np.float32)
        X_wf[i, 0] += float(label + 1) * np.sin((label + 1) * t)
        X_rh[i, 0] = 0.6 + 0.15 * float(label)
    return X_wf, X_rh, y


class TestMethodCArchitecture:
    def test_bahdanau_attention_shapes_and_normalization(self):
        attn = BahdanauTemporalAttention(hidden_dim=128, attn_dim=64)
        h = torch.randn(8, 45, 128)
        context, alphas = attn(h)
        assert context.shape == (8, 128)
        assert alphas.shape == (8, 45)
        sums = alphas.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(sums, np.ones(8), atol=1e-5)
        assert torch.all(alphas >= 0.0) and torch.all(alphas <= 1.0)

    def test_cnn_bilstm_attention_forward_pass(self):
        net = CNNBiLSTMAttention1D()
        x_wf = torch.randn(4, 1, 360)
        x_rh = torch.randn(4, 4)
        logits, alphas = net.forward_with_attention(x_wf, x_rh)
        assert logits.shape == (4, 5)
        assert alphas.shape == (4, 45)
        out = net(x_wf, x_rh)
        assert out.shape == (4, 5)
        total_params = sum(p.numel() for p in net.parameters())
        assert 200_000 <= total_params <= 400_000

    def test_gradient_flow_across_cnn_lstm_and_attention(self):
        net = CNNBiLSTMAttention1D()
        x_wf = torch.randn(6, 1, 360)
        x_rh = torch.randn(6, 4)
        y = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)
        logits = net(x_wf, x_rh)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


class TestMethodCClassifierContract:
    def test_fit_predict_and_proba_invariants(self):
        X_wf, X_rh, y = _generate_synthetic_dataset(n_samples=60)
        clf = MethodCHybridAttentionClassifier(epochs=3, batch_size=16, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)
        assert clf.is_fitted
        assert len(clf.loss_history) == 3

        proba = clf.predict_proba(X_wf[:10], X_rh[:10])
        assert proba.shape == (10, 5)
        assert proba.dtype == np.float32
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10), atol=1e-5)

        preds = clf.predict(X_wf[:10], X_rh[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_subset_classes_zero_masking(self):
        X_wf, X_rh, y = _generate_synthetic_dataset(n_samples=40, classes=(0, 2))
        clf = MethodCHybridAttentionClassifier(epochs=2, batch_size=16, device="cpu", random_state=42)
        clf.fit(X_wf, X_rh, y)
        proba = clf.predict_proba(X_wf[:8], X_rh[:8])
        assert np.all(proba[:, [1, 3, 4]] == 0.0)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(8), atol=1e-5)

    def test_extract_attention_weights_and_upsampling(self):
        X_wf, X_rh, y = _generate_synthetic_dataset(n_samples=20)
        clf = MethodCHybridAttentionClassifier(epochs=2, batch_size=10, device="cpu")
        clf.fit(X_wf, X_rh, y)

        attn_45 = clf.extract_attention_weights(X_wf[:5], X_rh[:5], upsample_to_360=False)
        assert attn_45.shape == (5, 45)
        np.testing.assert_allclose(attn_45.sum(axis=1), np.ones(5), atol=1e-5)

        attn_360 = clf.extract_attention_weights(X_wf[:5], X_rh[:5], upsample_to_360=True)
        assert attn_360.shape == (5, 360)
        np.testing.assert_allclose(attn_360.sum(axis=1), np.ones(5), atol=1e-5)

    def test_checkpoint_save_and_load_parity(self):
        X_wf, X_rh, y = _generate_synthetic_dataset(n_samples=30)
        clf = MethodCHybridAttentionClassifier(epochs=2, batch_size=15, device="cpu", random_state=7)
        clf.fit(X_wf, X_rh, y)
        orig_proba = clf.predict_proba(X_wf[:6], X_rh[:6])

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "method_c.pt")
            clf.save_checkpoint(ckpt_path)
            loaded = MethodCHybridAttentionClassifier.load_checkpoint(ckpt_path)
            loaded_proba = loaded.predict_proba(X_wf[:6], X_rh[:6])
            np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)
