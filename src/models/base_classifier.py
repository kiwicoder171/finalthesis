"""
Abstract Base Classifier for ECG Arrhythmia Rhythm Classification.
Adheres strictly to PROJECT.md Interface Contract 2 and ANSI/AAMI EC57 standards.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseECGClassifier(ABC):
    """
    Abstract Base Class for ECG beat classifiers under ANSI/AAMI EC57.
    All benchmark classification methods (Method A: GBDT, Method B: 1D-ResNet,
    Method C: CNN-BiLSTM-Attention) inherit from this unified interface.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([0, 1, 2, 3, 4], dtype=int)
        self.params: Dict[str, Any] = kwargs

    @staticmethod
    def validate_inputs(
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
        y: Optional[np.ndarray] = None,
        allow_empty: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Validate and standardize input shapes and types for ECG classifiers.

        Accepts:
        - X_waveform: shape (N, 1, 360), (N, 360), (N, 360, 1), or (360,)
        - X_rhythm: shape (N, 4) or (4,)
        - y: shape (N,), (N, 1), or None

        Returns:
        - X_wf: standardized shape (N, 1, 360), dtype float32
        - X_rh: standardized shape (N, 4), dtype float32
        - y_arr: standardized shape (N,), dtype int, or None
        """
        # --- Waveform Validation ---
        X_wf = np.asarray(X_waveform, dtype=np.float32)
        if X_wf.ndim == 1:
            if X_wf.shape[0] != 360:
                raise ValueError(f"Expected temporal dimension 360, got {X_wf.shape[0]}")
            X_wf = X_wf.reshape(1, 1, 360)
        elif X_wf.ndim == 2:
            if X_wf.shape[1] != 360:
                raise ValueError(f"Expected temporal dimension 360, got {X_wf.shape[1]}")
            X_wf = X_wf.reshape(-1, 1, 360)
        elif X_wf.ndim == 3:
            if X_wf.shape[1] == 1 and X_wf.shape[2] == 360:
                pass
            elif X_wf.shape[1] == 360 and X_wf.shape[2] == 1:
                X_wf = np.transpose(X_wf, (0, 2, 1))
            else:
                raise ValueError(f"Expected shape (N, 1, 360) or (N, 360), got {X_wf.shape}")
        else:
            raise ValueError(f"Invalid waveform ndim={X_wf.ndim}")

        # --- Rhythm Validation ---
        X_rh = np.asarray(X_rhythm, dtype=np.float32)
        if X_rh.ndim == 1:
            if X_rh.shape[0] != 4:
                raise ValueError(f"Expected 4 rhythm features, got {X_rh.shape[0]}")
            X_rh = X_rh.reshape(1, 4)
        elif X_rh.ndim == 2:
            if X_rh.shape[1] != 4:
                raise ValueError(f"Expected 4 rhythm features, got {X_rh.shape[1]}")
        else:
            raise ValueError(f"Invalid rhythm ndim={X_rh.ndim}")

        if len(X_wf) != len(X_rh):
            raise ValueError(f"Sample count mismatch: waveforms={len(X_wf)}, rhythm={len(X_rh)}")

        if not allow_empty and len(X_wf) == 0:
            raise ValueError("Input dataset is empty")

        if len(X_wf) > 0:
            if np.isnan(X_wf).any() or np.isinf(X_wf).any():
                raise ValueError("X_waveform contains NaN or Inf values")
            if np.isnan(X_rh).any() or np.isinf(X_rh).any():
                raise ValueError("X_rhythm contains NaN or Inf values")

        # --- Label Validation ---
        y_arr: Optional[np.ndarray] = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            if y_arr.ndim != 1:
                raise ValueError(f"Expected 1D label array, got ndim={y_arr.ndim}")
            if len(y_arr) != len(X_wf):
                raise ValueError(f"Sample count mismatch: labels={len(y_arr)}, waveforms={len(X_wf)}")
            if not np.issubdtype(y_arr.dtype, np.integer):
                try:
                    y_arr = y_arr.astype(int)
                except Exception:
                    raise ValueError("Labels must be integer convertible")
            if len(y_arr) > 0 and (np.min(y_arr) < 0 or np.max(y_arr) > 4):
                raise ValueError(f"Labels must be in range [0, 4], got {np.unique(y_arr)}")

        return X_wf, X_rh, y_arr

    @abstractmethod
    def fit(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "BaseECGClassifier":
        """
        Fit classifier on training heartbeats.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).
            y: Integer class labels in {0, 1, 2, 3, 4} of shape (N,).
            val_data: Optional validation tuple (X_val_wf, X_val_rh, y_val).

        Returns:
            self: The fitted BaseECGClassifier instance.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """
        Predict discrete ANSI/AAMI EC57 class labels.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).

        Returns:
            np.ndarray of shape (N,), dtype int, values in {0, 1, 2, 3, 4}.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """
        Predict calibrated class probabilities across all 5 AAMI classes.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).

        Returns:
            np.ndarray of shape (N, 5), dtype float32, rows sum to 1.0.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Destination file path for checkpoint serialization.
        """
        pass

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str) -> "BaseECGClassifier":
        """
        Load model checkpoint from disk.

        Args:
            path: Path to serialized checkpoint file.

        Returns:
            Loaded BaseECGClassifier instance.
        """
        pass
