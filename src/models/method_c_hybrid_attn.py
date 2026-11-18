"""
Method C: Hybrid 1D-CNN + Bidirectional LSTM with Bahdanau Temporal Attention & Rhythm Fusion.
Adheres strictly to PROJECT.md Interface Contract 2 and ANSI/AAMI EC57 standards.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_classifier import BaseECGClassifier
from src.models.method_b_resnet import MultiClassFocalLoss


class BahdanauTemporalAttention(nn.Module):
    """
    Bahdanau Additive Temporal Attention mechanism over BiLSTM hidden states.

    Given sequence H in R^(B, T, D):
      u_t = tanh(W_h * h_t + b_h)       in R^(B, T, A)
      e_t = v_a^T * u_t                 in R^(B, T)
      alpha_t = softmax(e_t, dim=1)     in R^(B, T), sum_t alpha_t = 1.0
      c = sum_t alpha_t * h_t           in R^(B, D)
    """

    def __init__(self, hidden_dim: int = 128, attn_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-pooled context vector and temporal attention weights.

        Args:
            h: BiLSTM sequence output of shape (B, T, D).

        Returns:
            context: Context vector of shape (B, D).
            alphas: Temporal attention weights of shape (B, T), rows sum to 1.0.
        """
        u = torch.tanh(self.W_h(h))                     # (B, T, A)
        scores = self.v_a(u).squeeze(-1)                # (B, T)
        alphas = F.softmax(scores, dim=-1)              # (B, T)
        context = torch.sum(h * alphas.unsqueeze(-1), dim=1)  # (B, D)
        return context, alphas


class CNNBiLSTMAttention1D(nn.Module):
    """
    Hybrid 1D-CNN + 2-Layer Bidirectional LSTM + Bahdanau Temporal Attention + Rhythm Fusion.

    Architecture:
    - 3-Layer 1D-CNN Morphological Frontend (360 samples -> 45 temporal tokens):
      * Block 1: Conv1d(1, 32, k=15, p=7) -> BN -> LeakyReLU(0.1) -> MaxPool1d(2) -> (B, 32, 180)
      * Block 2: Conv1d(32, 64, k=7, p=3) -> BN -> LeakyReLU(0.1) -> MaxPool1d(2) -> (B, 64, 90)
      * Block 3: Conv1d(64, 128, k=5, p=2) -> BN -> LeakyReLU(0.1) -> MaxPool1d(2) -> (B, 128, 45)
    - 2-Layer Bidirectional LSTM:
      * Input: (B, 45, 128) -> Hidden: 64 per direction -> Output: (B, 45, 128)
    - Bahdanau Temporal Attention:
      * Pools (B, 45, 128) -> Context c in (B, 128) + Attention weights alpha in (B, 45)
    - Auxiliary Rhythm MLP:
      * Linear(4, 32) -> LeakyReLU(0.1) -> Linear(32, 16) -> LeakyReLU(0.1) -> (B, 16)
    - Fusion & Classification Head:
      * Concat [c (128), h_rh (16)] -> (B, 144)
      * Linear(144, 64) -> LayerNorm(64) -> LeakyReLU(0.1) -> Dropout(p) -> Linear(64, 5)
    Total Parameters: ~274,000 (~274k).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        attn_dim: int = 64,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.attn_dim = attn_dim
        self.dropout_rate = dropout_rate

        # 1. 3-Layer 1D-CNN Frontend
        self.cnn_frontend = nn.Sequential(
            # Block 1: 360 -> 180
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2: 180 -> 90
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3: 90 -> 45
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 2. 2-Layer Bidirectional LSTM
        lstm_dropout = 0.2 if lstm_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        bilstm_out_dim = lstm_hidden * 2  # 128

        # 3. Bahdanau Temporal Attention
        self.attention = BahdanauTemporalAttention(
            hidden_dim=bilstm_out_dim,
            attn_dim=attn_dim,
        )

        # 4. Auxiliary Rhythm MLP (4D RR features -> 16D embedding)
        self.rhythm_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 5. Late Fusion & Classification Head
        fusion_dim = bilstm_out_dim + 16  # 144
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward_with_attention(
        self, x_wf: torch.Tensor, x_rh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both class logits and temporal attention weights.

        Args:
            x_wf: Waveform tensor of shape (B, 1, 360) or (B, 360).
            x_rh: Rhythm feature tensor of shape (B, 4).

        Returns:
            logits: Tensor of shape (B, 5).
            alphas: Temporal attention weights of shape (B, 45).
        """
        if x_wf.ndim == 2:
            x_wf = x_wf.unsqueeze(1)
        elif x_wf.ndim == 1:
            x_wf = x_wf.view(1, 1, -1)

        if x_rh.ndim == 1:
            x_rh = x_rh.view(1, -1)

        cnn_feat = self.cnn_frontend(x_wf)          # (B, 128, 45)
        seq_feat = cnn_feat.permute(0, 2, 1)        # (B, 45, 128)
        lstm_out, _ = self.bilstm(seq_feat)         # (B, 45, 128)
        context, alphas = self.attention(lstm_out)  # (B, 128), (B, 45)
        h_rh = self.rhythm_mlp(x_rh)                # (B, 16)
        fused = torch.cat([context, h_rh], dim=1)   # (B, 144)
        logits = self.classifier(fused)             # (B, 5)
        return logits, alphas

    def forward(self, x_wf: torch.Tensor, x_rh: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning class logits of shape (B, 5).
        """
        logits, _ = self.forward_with_attention(x_wf, x_rh)
        return logits


class MethodCHybridAttentionClassifier(BaseECGClassifier):
    """
    Method C: Hybrid 1D-CNN + BiLSTM + Bahdanau Temporal Attention Classifier.
    Adheres strictly to PROJECT.md Interface Contract 2 and ANSI/AAMI EC57 standards.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 2.0,
        alpha: Optional[Union[str, List[float], np.ndarray, torch.Tensor]] = "balanced",
        early_stopping_patience: int = 5,
        dropout_rate: float = 0.25,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        params: Dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "alpha": alpha,
            "early_stopping_patience": early_stopping_patience,
            "dropout_rate": dropout_rate,
            "max_grad_norm": max_grad_norm,
            "device": device,
            "random_state": random_state,
            **kwargs,
        }
        super().__init__(**params)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.alpha = alpha
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.max_grad_norm = max_grad_norm
        self.device_str = device
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_: Optional[CNNBiLSTMAttention1D] = None
        self.fitted_classes_: np.ndarray = np.array([], dtype=int)
        self.loss_history_: List[float] = []
        self.device_: torch.device = self._resolve_device(device)

    @property
    def loss_history(self) -> List[float]:
        """Return list of average training losses per epoch."""
        return self.loss_history_

    @staticmethod
    def _resolve_device(device_str: Optional[str]) -> torch.device:
        if device_str is not None:
            try:
                dev = torch.device(device_str)
                if dev.type == "cuda" and not torch.cuda.is_available():
                    dev = None
                elif dev.type == "mps" and not torch.backends.mps.is_available():
                    dev = None
                elif dev.type not in ("cpu", "cuda", "mps"):
                    dev = None
                else:
                    _ = torch.zeros(1, device=dev)
                    return dev
            except Exception:
                pass
        if torch.backends.mps.is_available():
            try:
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    def _compute_alpha_weights(self, y: np.ndarray) -> Optional[torch.Tensor]:
        if self.alpha is None or self.alpha is False or self.alpha == "none":
            return None
        if isinstance(self.alpha, str) and self.alpha == "balanced":
            counts = np.bincount(y, minlength=5).astype(np.float32)
            n_samples = float(len(y))
            n_classes = 5.0
            weights = np.zeros(5, dtype=np.float32)
            for c in range(5):
                if counts[c] > 0:
                    weights[c] = n_samples / (n_classes * counts[c])
                else:
                    weights[c] = 1.0
            weights = weights / (weights.mean() + 1e-8)
            return torch.tensor(weights, dtype=torch.float32)
        elif isinstance(self.alpha, (list, tuple, np.ndarray, torch.Tensor)):
            arr = np.asarray(self.alpha, dtype=np.float32)
            if len(arr) != 5:
                raise ValueError(f"Expected alpha of length 5, got {len(arr)}")
            return torch.tensor(arr, dtype=torch.float32)
        raise ValueError(f"Invalid alpha specification: {self.alpha}")

    def fit(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "MethodCHybridAttentionClassifier":
        """
        Fit CNN-BiLSTM-Attention classifier on training heartbeats.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).
            y: Integer class labels in {0, 1, 2, 3, 4} of shape (N,).
            val_data: Optional validation tuple (X_val_wf, X_val_rh, y_val).

        Returns:
            self: The fitted MethodCHybridAttentionClassifier instance.
        """
        X_wf, X_rh, y_arr = self.validate_inputs(X_waveform, X_rhythm, y)
        assert y_arr is not None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.fitted_classes_ = np.unique(y_arr)
        self.loss_history_ = []

        # Single-class edge case
        if len(self.fitted_classes_) == 1:
            self.model_ = CNNBiLSTMAttention1D(dropout_rate=self.dropout_rate).to(self.device_)
            self.model_.eval()
            self.is_fitted = True
            self.loss_history_ = [0.0]
            return self

        self.model_ = CNNBiLSTMAttention1D(dropout_rate=self.dropout_rate).to(self.device_)
        alpha_tensor = self._compute_alpha_weights(y_arr)
        criterion = MultiClassFocalLoss(alpha=alpha_tensor, gamma=self.gamma)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_ds = TensorDataset(
            torch.from_numpy(X_wf),
            torch.from_numpy(X_rh),
            torch.from_numpy(y_arr).long(),
        )
        batch_size = min(self.batch_size, max(len(train_ds), 1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        val_loader = None
        val_ds = None
        if val_data is not None:
            v_wf, v_rh, v_y = self.validate_inputs(val_data[0], val_data[1], val_data[2])
            assert v_y is not None
            val_ds = TensorDataset(
                torch.from_numpy(v_wf),
                torch.from_numpy(v_rh),
                torch.from_numpy(v_y).long(),
            )
            v_batch_size = min(self.batch_size, max(len(val_ds), 1))
            val_loader = DataLoader(val_ds, batch_size=v_batch_size, shuffle=False)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0
            for b_wf, b_rh, b_y in train_loader:
                b_wf = b_wf.to(self.device_)
                b_rh = b_rh.to(self.device_)
                b_y = b_y.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(b_wf, b_rh)
                loss = criterion(logits, b_y)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.max_grad_norm)
                optimizer.step()
                train_loss += loss.item() * len(b_y)
            train_loss /= max(len(train_ds), 1)
            self.loss_history_.append(float(train_loss))

            if val_loader is not None and val_ds is not None and len(val_ds) > 0:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for b_wf, b_rh, b_y in val_loader:
                        b_wf = b_wf.to(self.device_)
                        b_rh = b_rh.to(self.device_)
                        b_y = b_y.to(self.device_)
                        logits = self.model_(b_wf, b_rh)
                        loss = criterion(logits, b_y)
                        val_loss += loss.item() * len(b_y)
                val_loss /= len(val_ds)
                eval_loss = val_loss
            else:
                eval_loss = train_loss

            if eval_loss < best_loss - 1e-4:
                best_loss = eval_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """
        Predict calibrated class probabilities across all 5 AAMI classes.

        Returns:
            np.ndarray of shape (N, 5), dtype float32, rows sum to 1.0.
        """
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")

        X_wf, X_rh, _ = self.validate_inputs(X_waveform, X_rhythm, allow_empty=True)
        N = len(X_wf)
        if N == 0:
            return np.zeros((0, 5), dtype=np.float32)

        if len(self.fitted_classes_) == 1:
            full_proba = np.zeros((N, 5), dtype=np.float32)
            c_int = int(self.fitted_classes_[0])
            full_proba[:, c_int] = 1.0
            return full_proba

        self.model_.eval()
        proba_list = []
        batch_size = max(128, self.batch_size)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                b_wf = torch.from_numpy(X_wf[i : i + batch_size]).to(self.device_)
                b_rh = torch.from_numpy(X_rh[i : i + batch_size]).to(self.device_)
                logits = self.model_(b_wf, b_rh)
                if len(self.fitted_classes_) < 5:
                    unobserved = [c for c in range(5) if c not in self.fitted_classes_]
                    logits[:, unobserved] = -1e9
                probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
                proba_list.append(probs)

        full_proba = np.vstack(proba_list)

        if len(self.fitted_classes_) < 5:
            mask = np.zeros(5, dtype=bool)
            mask[self.fitted_classes_] = True
            full_proba[:, ~mask] = 0.0

        full_proba = np.clip(full_proba, 0.0, 1.0)
        row_sums = full_proba.sum(axis=1, keepdims=True)
        zero_mask = (row_sums.squeeze(-1) == 0)
        nonzero_sums = np.where(row_sums > 0, row_sums, 1.0)
        full_proba = full_proba / nonzero_sums
        if np.any(zero_mask):
            for c in self.fitted_classes_:
                full_proba[zero_mask, c] = 1.0 / len(self.fitted_classes_)

        return full_proba.astype(np.float32)

    def predict(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """
        Predict discrete ANSI/AAMI EC57 class labels in {0, 1, 2, 3, 4}.
        """
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X_wf, X_rh, _ = self.validate_inputs(X_waveform, X_rhythm, allow_empty=True)
        N = len(X_wf)
        if N == 0:
            return np.zeros(0, dtype=int)

        if len(self.fitted_classes_) == 1:
            return np.full(N, int(self.fitted_classes_[0]), dtype=int)

        proba = self.predict_proba(X_wf, X_rh)
        return np.argmax(proba, axis=1).astype(int)

    def extract_attention_weights(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
        upsample_to_360: bool = False,
    ) -> np.ndarray:
        """
        Extract temporal attention weights mapping to P-QRS-T wave segments.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).
            upsample_to_360: If True, linearly interpolates the 45-step attention
                distribution back to the 360-sample signal domain (rows sum to 1.0).

        Returns:
            Attention weights of shape (N, 45) or (N, 360), dtype float32, rows sum to 1.0.
        """
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before extract_attention_weights().")

        X_wf, X_rh, _ = self.validate_inputs(X_waveform, X_rhythm, allow_empty=True)
        N = len(X_wf)
        target_len = 360 if upsample_to_360 else 45
        if N == 0:
            return np.zeros((0, target_len), dtype=np.float32)

        self.model_.eval()
        attn_list = []
        batch_size = max(128, self.batch_size)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                b_wf = torch.from_numpy(X_wf[i : i + batch_size]).to(self.device_)
                b_rh = torch.from_numpy(X_rh[i : i + batch_size]).to(self.device_)
                _, alphas = self.model_.forward_with_attention(b_wf, b_rh)
                if upsample_to_360:
                    alphas_up = F.interpolate(
                        alphas.unsqueeze(1),
                        size=360,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)
                    alphas_up = alphas_up / (alphas_up.sum(dim=-1, keepdim=True) + 1e-12)
                    attn_list.append(alphas_up.cpu().numpy().astype(np.float32))
                else:
                    attn_list.append(alphas.cpu().numpy().astype(np.float32))

        return np.vstack(attn_list)

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint to disk.
        """
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Cannot save unfitted model checkpoint.")

        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model_.state_dict(),
            "fitted_classes": self.fitted_classes_.tolist(),
            "params": self.params,
            "is_fitted": self.is_fitted,
            "dropout_rate": self.dropout_rate,
            "classes": self.classes_.tolist(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MethodCHybridAttentionClassifier":
        """
        Load model checkpoint from disk with cross-device resilience.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        params = checkpoint.get("params", {}).copy()

        req_dev_str = params.get("device")
        if req_dev_str is not None:
            try:
                test_dev = torch.device(req_dev_str)
                if test_dev.type == "cuda" and not torch.cuda.is_available():
                    params["device"] = None
                elif test_dev.type == "mps" and not torch.backends.mps.is_available():
                    params["device"] = None
                elif test_dev.type not in ("cpu", "cuda", "mps"):
                    params["device"] = None
                else:
                    _ = torch.zeros(1, device=test_dev)
            except Exception:
                params["device"] = None

        instance = cls(**params)
        instance.fitted_classes_ = np.asarray(checkpoint["fitted_classes"], dtype=int)
        instance.is_fitted = bool(checkpoint["is_fitted"])
        dropout_rate = checkpoint.get("dropout_rate", params.get("dropout_rate", 0.25))
        instance.model_ = CNNBiLSTMAttention1D(dropout_rate=dropout_rate)
        instance.model_.load_state_dict(checkpoint["model_state_dict"])
        instance.model_.to(instance.device_)
        instance.model_.eval()
        return instance
