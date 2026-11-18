"""
ECG Signal Preprocessor Module.

Provides clinical-grade digital signal processing for ECG signals:
- Butterworth zero-phase bandpass filtering (0.5 - 45.0 Hz) to eliminate
  respiratory baseline wander and high-frequency EMG noise.
- Linear and polynomial baseline detrending.
- Polyphase / Fourier resampling to arbitrary target frequencies (e.g., 360 Hz standard).
- Z-score standardization: (x - mu) / (sigma + eps).
- Multi-lead signal support.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal


class ECGPreprocessor:
    """Configurable digital signal preprocessor for ECG waveforms."""

    def __init__(
        self,
        target_fs: float = 360.0,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        filter_order: int = 4,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_fs: Target sampling frequency in Hz (default: 360.0 Hz for MIT-BIH parity).
            lowcut: Lower cutoff frequency in Hz for baseline suppression.
            highcut: Upper cutoff frequency in Hz for high-frequency noise suppression.
            filter_order: Order of the Butterworth filter.
            normalize: Whether to apply Z-score normalization.
            eps: Epsilon to avoid division by zero during normalization.
        """
        self.target_fs = float(target_fs)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.filter_order = int(filter_order)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def bandpass_filter(
        self,
        ecg_signal: np.ndarray,
        fs: float,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        order: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply zero-phase Butterworth bandpass filter.

        Args:
            ecg_signal: 1D or 2D array of ECG voltage.
            fs: Original sampling frequency in Hz.
            lowcut: Low cutoff frequency (defaults to self.lowcut).
            highcut: High cutoff frequency (defaults to self.highcut).
            order: Filter order (defaults to self.filter_order).

        Returns:
            Filtered ECG signal with identical shape and zero phase shift.
        """
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive (> 0 Hz), got {fs} Hz")

        low = self.lowcut if lowcut is None else float(lowcut)
        high = self.highcut if highcut is None else float(highcut)
        ord_val = self.filter_order if order is None else int(order)

        if low <= 0:
            raise ValueError(f"Low cutoff frequency must be positive (> 0 Hz), got {low} Hz")

        nyquist = 0.5 * fs
        if nyquist <= low:
            raise ValueError(f"Sampling frequency {fs} Hz is too low for lowcut {low} Hz (Nyquist: {nyquist} Hz)")

        if high <= low:
            raise ValueError(f"High cutoff frequency ({high} Hz) must be strictly greater than low cutoff frequency ({low} Hz)")

        # Ensure highcut is strictly below Nyquist frequency and strictly above lowcut:
        # If high >= nyquist, clamp to 0.90 * nyquist if safe, or the arithmetic midpoint between low and nyquist.
        if high >= nyquist:
            high = max(0.90 * nyquist, low + 0.5 * (nyquist - low))

        # Design Butterworth filter
        low_norm = low / nyquist
        high_norm = high / nyquist

        b, a = signal.butter(ord_val, [low_norm, high_norm], btype="bandpass")

        sig_arr = np.asarray(ecg_signal, dtype=np.float64)

        # Early return for empty signal
        if sig_arr.size == 0:
            return np.zeros_like(sig_arr)

        # Zero-phase forward-backward filtering
        # Check signal length vs pad length requirement for filtfilt
        min_len = 3 * max(len(a), len(b))
        is_1d = sig_arr.ndim == 1

        if is_1d:
            if len(sig_arr) <= min_len:
                # Signal too short for filtfilt pad length, return detrended
                return signal.detrend(sig_arr)
            filtered = signal.filtfilt(b, a, sig_arr)
        else:
            # Multi-channel signal (N_samples, N_channels) or (N_channels, N_samples)
            # Filter along the longest axis
            axis = 0 if sig_arr.shape[0] >= sig_arr.shape[1] else 1
            if sig_arr.shape[axis] <= min_len:
                return signal.detrend(sig_arr, axis=axis)
            filtered = signal.filtfilt(b, a, sig_arr, axis=axis)

        return filtered

    def detrend(self, ecg_signal: np.ndarray, axis: int = 0) -> np.ndarray:
        """Remove linear trend from ECG signal."""
        sig_arr = np.asarray(ecg_signal, dtype=np.float64)
        if sig_arr.size == 0:
            return np.zeros_like(sig_arr)
        return signal.detrend(sig_arr, axis=axis)

    def resample(
        self,
        ecg_signal: np.ndarray,
        orig_fs: float,
        target_fs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Resample ECG signal from orig_fs to target_fs.

        Args:
            ecg_signal: 1D or 2D ECG signal array.
            orig_fs: Source sampling rate in Hz.
            target_fs: Destination sampling rate in Hz (defaults to self.target_fs).

        Returns:
            Resampled ECG array.
        """
        target = self.target_fs if target_fs is None else float(target_fs)
        if np.isclose(orig_fs, target, atol=1e-3):
            return ecg_signal.copy()

        sig_arr = np.asarray(ecg_signal, dtype=np.float64)
        if sig_arr.size == 0:
            return np.zeros_like(sig_arr)

        is_1d = sig_arr.ndim == 1

        if is_1d:
            num_samples = int(round(len(sig_arr) * (target / orig_fs)))
            if num_samples <= 0:
                return np.zeros((0,), dtype=np.float64)
            return signal.resample(sig_arr, num_samples)
        else:
            axis = 0 if sig_arr.shape[0] >= sig_arr.shape[1] else 1
            orig_len = sig_arr.shape[axis]
            num_samples = int(round(orig_len * (target / orig_fs)))
            if num_samples <= 0:
                shape = list(sig_arr.shape)
                shape[axis] = 0
                return np.zeros(shape, dtype=np.float64)
            return signal.resample(sig_arr, num_samples, axis=axis)

    def z_score_normalize(
        self, ecg_signal: np.ndarray, axis: Optional[int] = None
    ) -> np.ndarray:
        """
        Standardize ECG signal using Z-score: (x - mean) / (std + eps).
        """
        sig_arr = np.asarray(ecg_signal, dtype=np.float32)
        if sig_arr.size == 0:
            return np.zeros_like(sig_arr)

        if axis is None:
            mean = np.mean(sig_arr)
            std = np.std(sig_arr)
        else:
            mean = np.mean(sig_arr, axis=axis, keepdims=True)
            std = np.std(sig_arr, axis=axis, keepdims=True)

        return (sig_arr - mean) / (std + self.eps)

    def preprocess(
        self,
        ecg_signal: np.ndarray,
        fs: float,
        target_fs: Optional[float] = None,
    ) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Butterworth 0.5-45 Hz bandpass zero-phase filtering.
        2. Resampling to target frequency if needed.
        3. Z-score normalization if enabled.

        Args:
            ecg_signal: Raw input ECG array.
            fs: Original sampling frequency.
            target_fs: Target sampling frequency (optional override).

        Returns:
            Preprocessed ECG array as float32.
        """
        sig_arr = np.asarray(ecg_signal, dtype=np.float32)
        if sig_arr.size == 0:
            return np.zeros_like(sig_arr)

        tgt = self.target_fs if target_fs is None else float(target_fs)

        # 1. Bandpass filter
        filtered = self.bandpass_filter(sig_arr, fs=fs)

        # 2. Resample if necessary
        resampled = self.resample(filtered, orig_fs=fs, target_fs=tgt)

        # 3. Normalize if enabled
        if self.normalize:
            normalized = self.z_score_normalize(resampled)
            return normalized.astype(np.float32)

        return resampled.astype(np.float32)


def resample_indices(
    sample_indices: Union[np.ndarray, list],
    orig_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Scale sample annotation indices from orig_fs to target_fs.

    Args:
        sample_indices: Array or list of sample indices.
        orig_fs: Original sampling frequency in Hz.
        target_fs: Target sampling frequency in Hz.

    Returns:
        Integer array of scaled sample indices.
    """
    arr = np.asarray(sample_indices, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros((0,), dtype=int)
    if np.isclose(orig_fs, target_fs, atol=1e-3):
        return arr.astype(int)
    factor = target_fs / orig_fs
    return np.round(arr * factor).astype(int)
