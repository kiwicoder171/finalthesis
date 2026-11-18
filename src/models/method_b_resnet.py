"""
Method B: Deep 1D Residual Convolutional Neural Network (1D-ResNet) with Rhythm Fusion.
Adheres strictly to PROJECT.md Contract 2 and ANSI/AAMI EC57 standards.
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


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with 2 Conv1d layers, BatchNorm, ReLU, and Dropout.
    Supports stride downsampling and 1x1 conv channel projection in skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class ECGResNet1D(nn.Module):
    """
    Deep 1D-ResNet for ECG Beat Classification with Auxiliary Rhythm Fusion.

    Architecture:
    - Stem Conv: Conv1d(1, 64, k=15, s=2, p=7) -> BN -> ReLU -> MaxPool1d(3, s=2, p=1) -> (B, 64, 90)
    - 8 Residual Blocks across 3 stages [2, 3, 3] with channels 64 -> 128 -> 256:
      * Stage 1: 2 blocks @ 64 channels, k=7, s=1, dropout=0.1 -> (B, 64, 90)
      * Stage 2: 3 blocks @ 128 channels, k=5, s=2 on first block, dropout=0.15 -> (B, 128, 45)
      * Stage 3: 3 blocks @ 256 channels, k=3, s=2 on first block, dropout=0.2 -> (B, 256, 23)
    - Global Average Pooling: AdaptiveAvgPool1d(1) -> (B, 256)
    - Auxiliary Rhythm MLP: Linear(4, 32) -> ReLU -> Linear(32, 32) -> ReLU -> (B, 32)
    - Fusion: Concatenate [h_wf, h_rh] -> (B, 288)
    - Classification Head: Linear(288, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 5) -> (B, 5)
    Total Parameters: 1,714,565 (~1.71M).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # 1. Stem Conv Block
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 2. 8 Residual Blocks across channels 64 -> 128 -> 256
        # Stage 1: 2 blocks @ 64 channels, k=7, s=1
        self.stage1 = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=7, stride=1, dropout=0.1),
            ResidualBlock1D(64, 64, kernel_size=7, stride=1, dropout=0.1),
        )
        # Stage 2: 3 blocks @ 128 channels, k=5, s=2 on first block
        self.stage2 = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=5, stride=2, dropout=0.15),
            ResidualBlock1D(128, 128, kernel_size=5, stride=1, dropout=0.15),
            ResidualBlock1D(128, 128, kernel_size=5, stride=1, dropout=0.15),
        )
        # Stage 3: 3 blocks @ 256 channels, k=3, s=2 on first block
        self.stage3 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=3, stride=2, dropout=0.2),
            ResidualBlock1D(256, 256, kernel_size=3, stride=1, dropout=0.2),
            ResidualBlock1D(256, 256, kernel_size=3, stride=1, dropout=0.2),
        )

        # 3. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 4. Auxiliary Rhythm MLP
        self.rhythm_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # 5. Fusion & Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_wf: torch.Tensor, x_rh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D-ResNet with Rhythm Fusion.

        Args:
            x_wf: Waveform tensor of shape (B, 1, 360) or (B, 360).
            x_rh: Rhythm feature tensor of shape (B, 4).

        Returns:
            Logits tensor of shape (B, 5).
        """
        if x_wf.ndim == 2:
            x_wf = x_wf.unsqueeze(1)
        elif x_wf.ndim == 1:
            x_wf = x_wf.view(1, 1, -1)

        if x_rh.ndim == 1:
            x_rh = x_rh.view(1, -1)

        h = self.stem(x_wf)                      # (B, 64, 90)
        h = self.stage1(h)                       # (B, 64, 90)
        h = self.stage2(h)                       # (B, 128, 45)
        h = self.stage3(h)                       # (B, 256, 23)
        h_wf = torch.flatten(self.gap(h), 1)     # (B, 256)

        h_rh = self.rhythm_mlp(x_rh)             # (B, 32)
        fused = torch.cat([h_wf, h_rh], dim=1)   # (B, 288)
        logits = self.classifier(fused)          # (B, 5)
        return logits


class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with focusing parameter gamma and class weights alpha.
    Loss = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Output logits of shape (B, C).
            targets: Target class indices of shape (B,).

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum', else (B,) tensor.
        """
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = torch.clamp(1.0 - pt, min=0.0, max=1.0) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_buf = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha_buf.gather(dim=-1, index=targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MethodBResNetClassifier(BaseECGClassifier):
    """
    Method B: Deep 1D-ResNet Classifier with Auxiliary Rhythm MLP Fusion.
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
        dropout_rate: float = 0.2,
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
        self.device_str = device
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_: Optional[ECGResNet1D] = None
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
    ) -> "MethodBResNetClassifier":
        """
        Fit 1D-ResNet classifier on training heartbeats.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).
            y: Integer class labels in {0, 1, 2, 3, 4} of shape (N,).
            val_data: Optional validation tuple (X_val_wf, X_val_rh, y_val).

        Returns:
            self: The fitted MethodBResNetClassifier instance.
        """
        X_wf, X_rh, y_arr = self.validate_inputs(X_waveform, X_rhythm, y)
        assert y_arr is not None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.fitted_classes_ = np.unique(y_arr)
        self.loss_history_ = []

        # Single-class edge case: instantiate model and record single fitted class
        if len(self.fitted_classes_) == 1:
            self.model_ = ECGResNet1D(dropout_rate=self.dropout_rate).to(self.device_)
            self.model_.eval()
            self.is_fitted = True
            self.loss_history_ = [0.0]
            return self

        self.model_ = ECGResNet1D(dropout_rate=self.dropout_rate).to(self.device_)
        alpha_tensor = self._compute_alpha_weights(y_arr)
        criterion = MultiClassFocalLoss(alpha=alpha_tensor, gamma=self.gamma)
        optimizer = torch.optim.Adam(
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
                optimizer.step()
                train_loss += loss.item() * len(b_y)
            train_loss /= max(len(train_ds), 1)
            self.loss_history_.append(float(train_loss))

            # Validation / early stopping evaluation
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

        Ensures full (N, 5) float32 matrix with unobserved classes mapped to 0.0
        and each row strictly summing to 1.0.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).

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

        # Subset classes projection: zero out unobserved classes and renormalize
        if len(self.fitted_classes_) < 5:
            mask = np.zeros(5, dtype=bool)
            mask[self.fitted_classes_] = True
            full_proba[:, ~mask] = 0.0

        full_proba = np.clip(full_proba, 0.0, 1.0)
        row_sums = full_proba.sum(axis=1, keepdims=True)
        zero_mask = (row_sums.squeeze(-1) == 0)
        # Safe division
        nonzero_sums = np.where(row_sums > 0, row_sums, 1.0)
        full_proba = full_proba / nonzero_sums
        if np.any(zero_mask):
            # Assign uniform probability mass strictly across fitted classes for underflow rows
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

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).

        Returns:
            np.ndarray of shape (N,), dtype int, values in {0, 1, 2, 3, 4}.
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

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Destination file path for checkpoint serialization.
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
    def load_checkpoint(cls, path: str) -> "MethodBResNetClassifier":
        """
        Load model checkpoint from disk. Handles PyTorch 2.6+ weights_only policy.

        Args:
            path: Path to serialized checkpoint file.

        Returns:
            Loaded MethodBResNetClassifier instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        params = checkpoint.get("params", {}).copy()

        # Cross-device resilience: verify stored device availability on current host
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
        dropout_rate = checkpoint.get("dropout_rate", params.get("dropout_rate", 0.2))
        instance.model_ = ECGResNet1D(dropout_rate=dropout_rate)
        instance.model_.load_state_dict(checkpoint["model_state_dict"])
        instance.model_.to(instance.device_)
        instance.model_.eval()
        return instance
