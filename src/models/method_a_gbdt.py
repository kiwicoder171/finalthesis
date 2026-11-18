"""
Method A: Engineered Time-Frequency Features + Gradient Boosted Decision Trees.
Adheres to PROJECT.md Contract 2 and ANSI/AAMI EC57 standards.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pywt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.base_classifier import BaseECGClassifier


def extract_features(
    X_waveform: np.ndarray,
    X_rhythm: np.ndarray,
) -> np.ndarray:
    """
    Extract exact 42-dimensional feature vectors for heartbeats.

    Partitioned into three self-contained groups:
    1. Rhythm & Timing Dynamics (8 dimensions, f0 - f7):
       RR_pre, RR_post, delta_RR, RR_ratio, RR_mean, RR_pre_norm, RR_post_norm, ln(max(RR_ratio, eps))
    2. Morphological & Pan-Tompkins Metrics (14 dimensions, f8 - f21):
       R-peak amplitude, Q-valley nadir, S-valley nadir, QRS peak-to-peak, QRS width,
       QRS energy, total energy, QRS energy ratio, pre-R energy, post-R energy,
       pre-to-post energy ratio, waveform skewness, waveform kurtosis, max QRS slope
    3. DWT Daubechies 4 Wavelet Sub-bands (20 dimensions, f22 - f41):
       4-level db4 decomposition yielding 5 sub-bands (cA4, cD4, cD3, cD2, cD1).
       For each sub-band: [energy, variance, skewness, Shannon entropy]

    Args:
        X_waveform: Heartbeat waveforms, shape (N, 1, 360) or (N, 360) or (360,).
        X_rhythm: Rhythm features, shape (N, 4) or (4,).

    Returns:
        np.ndarray of shape (N, 42), dtype float32.
    """
    X_wf, X_rh, _ = BaseECGClassifier.validate_inputs(X_waveform, X_rhythm, allow_empty=True)
    N = len(X_wf)
    if N == 0:
        return np.zeros((0, 42), dtype=np.float32)

    X_2d = X_wf[:, 0, :].astype(np.float64)  # (N, 360)
    R_2d = X_rh.astype(np.float64)           # (N, 4)
    features = np.zeros((N, 42), dtype=np.float32)
    eps = 1e-8

    for i in range(N):
        x = X_2d[i]
        rh = R_2d[i]

        # 1. Rhythm & Timing Dynamics (8 features: f0 - f7)
        rr_pre, rr_post, delta_rr, rr_ratio = rh[0], rh[1], rh[2], rh[3]
        rr_mean = 0.5 * (rr_pre + rr_post)
        rr_pre_norm = rr_pre / (rr_mean + eps)
        rr_post_norm = rr_post / (rr_mean + eps)
        log_rr_ratio = np.log(max(rr_ratio, eps))
        f_rhythm = [
            rr_pre, rr_post, delta_rr, rr_ratio,
            rr_mean, rr_pre_norm, rr_post_norm, log_rr_ratio,
        ]

        # 2. Morphological & Pan-Tompkins Metrics (14 features: f8 - f21)
        r_idx = 120
        qrs_start = max(0, r_idx - 25)
        qrs_end = min(len(x), r_idx + 25)
        qrs_seg = x[qrs_start:qrs_end]
        r_amp = x[r_idx]
        q_val = np.min(x[qrs_start:r_idx])
        s_val = np.min(x[r_idx:qrs_end])
        qrs_p2p = np.max(qrs_seg) - np.min(qrs_seg)
        qrs_width = float(np.sum(np.abs(qrs_seg - np.mean(qrs_seg)) > (0.2 * qrs_p2p + eps)))
        qrs_energy = np.sum(qrs_seg ** 2)
        tot_energy = np.sum(x ** 2)
        energy_ratio = qrs_energy / (tot_energy + eps)
        pre_energy = np.sum(x[:r_idx] ** 2)
        post_energy = np.sum(x[r_idx:] ** 2)
        pre_post_ratio = pre_energy / (post_energy + eps)
        x_std = np.std(x)
        skew = float(stats.skew(x)) if x_std >= eps else 0.0
        kurt = float(stats.kurtosis(x)) if x_std >= eps else 0.0
        diff_qrs = np.diff(qrs_seg)
        max_slope = float(np.max(np.abs(diff_qrs))) if len(diff_qrs) > 0 else 0.0
        f_morph = [
            r_amp, q_val, s_val, qrs_p2p, qrs_width,
            qrs_energy, tot_energy, energy_ratio,
            pre_energy, post_energy, pre_post_ratio,
            skew, kurt, max_slope,
        ]

        # 3. DWT Daubechies 4 Wavelet Sub-bands (20 features: f22 - f41)
        # 5 sub-bands: cA4, cD4, cD3, cD2, cD1
        coeffs = pywt.wavedec(x, "db4", level=4)
        f_wavelet = []
        for sb in coeffs:
            sb_energy = np.sum(sb ** 2)
            sb_var = np.var(sb)
            sb_std = np.std(sb)
            sb_skew = float(stats.skew(sb)) if sb_std >= eps else 0.0
            p = (sb ** 2) / (sb_energy + eps)
            sb_entropy = -np.sum(p * np.log2(p + eps))
            f_wavelet.extend([sb_energy, sb_var, sb_skew, sb_entropy])

        features[i] = f_rhythm + f_morph + f_wavelet

    return features


class MethodAGBDTClassifier(BaseECGClassifier):
    """
    Method A GBDT Classifier using HistGradientBoostingClassifier.
    Fulfills Contract 2 unified ECG rhythm classification interface.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        params: Dict[str, Any] = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "random_state": random_state,
            **kwargs,
        }
        super().__init__(**params)

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_: Optional[HistGradientBoostingClassifier] = None
        self.fitted_classes_: np.ndarray = np.array([], dtype=int)

    def extract_features(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """Extract 42-dimensional feature vector for given waveforms and rhythm metrics."""
        return extract_features(X_waveform, X_rhythm)

    def fit(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "MethodAGBDTClassifier":
        """
        Fit GBDT classifier on training heartbeats.

        Args:
            X_waveform: Heartbeat waveforms of shape (N, 1, 360) or (N, 360).
            X_rhythm: Rhythm features of shape (N, 4).
            y: Integer class labels in {0, 1, 2, 3, 4} of shape (N,).
            val_data: Optional validation tuple (X_val_wf, X_val_rh, y_val).

        Returns:
            self: Fitted MethodAGBDTClassifier instance.
        """
        X_wf, X_rh, y_arr = self.validate_inputs(X_waveform, X_rhythm, y)
        X_feat = self.extract_features(X_wf, X_rh)

        self.model_ = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs,
        )

        fit_kwargs: Dict[str, Any] = {}
        if val_data is not None:
            X_val_wf, X_val_rh, y_val_arr = self.validate_inputs(
                val_data[0], val_data[1], val_data[2]
            )
            fit_kwargs["X_val"] = self.extract_features(X_val_wf, X_val_rh)
            fit_kwargs["y_val"] = y_val_arr

        self.model_.fit(X_feat, y_arr, **fit_kwargs)
        self.fitted_classes_ = np.unique(y_arr)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X_waveform: np.ndarray,
        X_rhythm: np.ndarray,
    ) -> np.ndarray:
        """
        Predict calibrated class probabilities across all 5 AAMI EC57 classes.

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

        X_feat = self.extract_features(X_wf, X_rh)
        full_proba = np.zeros((N, 5), dtype=np.float32)

        if len(self.fitted_classes_) == 1:
            # Single-class edge case: assign all probability mass to the fitted class
            target_class = int(self.fitted_classes_[0])
            if 0 <= target_class < 5:
                full_proba[:, target_class] = 1.0
            else:
                full_proba[:, 0] = 1.0
        else:
            raw_proba = self.model_.predict_proba(X_feat)
            for j, c in enumerate(self.model_.classes_):
                c_int = int(c)
                if 0 <= c_int < 5:
                    full_proba[:, c_int] = raw_proba[:, j]

            row_sums = full_proba.sum(axis=1, keepdims=True)
            full_proba = np.where(row_sums > 0, full_proba / row_sums, 0.2).astype(np.float32)

        return full_proba

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
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X_wf, X_rh, _ = self.validate_inputs(X_waveform, X_rhythm, allow_empty=True)
        N = len(X_wf)
        if N == 0:
            return np.zeros(0, dtype=int)

        proba = self.predict_proba(X_wf, X_rh)
        return np.argmax(proba, axis=1).astype(int)

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Target file path. Parent directories are created if missing.
        """
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Cannot save unfitted model checkpoint.")

        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        payload = {
            "model": self.model_,
            "fitted_classes": self.fitted_classes_,
            "params": self.params,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MethodAGBDTClassifier":
        """
        Load model checkpoint from disk.

        Args:
            path: Checkpoint file path.

        Returns:
            Restored MethodAGBDTClassifier instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        payload = joblib.load(path)
        params = payload.get("params", {})
        instance = cls(**params)
        instance.model_ = payload["model"]
        instance.fitted_classes_ = payload["fitted_classes"]
        instance.is_fitted = payload["is_fitted"]
        return instance
