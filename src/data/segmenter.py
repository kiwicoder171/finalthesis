"""
ECG Beat Segmentation and ANSI/AAMI EC57 Mapping Module.

Implements:
- ANSI/AAMI EC57 5-Class Beat Superclass Mapping (N, S, V, F, Q).
- Non-beat annotation filtering (rhythm markers, signal noise, wave landmarks).
- R-peak locked 360-sample windowing (120 pre-R, 240 post-R samples).
- 4-dimensional rhythm dynamics extraction: [RR_pre, RR_post, delta_RR, RR_ratio].
- Strict compliance with PROJECT.md Interface Contract 1.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.data.preprocessor import ECGPreprocessor, resample_indices


# ANSI/AAMI EC57 Beat Mapping Dictionary
# Maps 19 raw PhysioNet / WFDB annotation characters to 5 clinical superclasses:
# N: Normal / Non-ectopic
# S: Supraventricular ectopic (SVEB)
# V: Ventricular ectopic (VEB)
# F: Fusion beat
# Q: Unknown / Paced / Unclassifiable
AAMI_BEAT_MAP: Dict[str, str] = {
    # Class N: Normal / Bundle Branch Blocks / Escape beats
    "N": "N",  # Normal sinus beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "B": "N",  # Bundle branch block beat (unspecified)
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    "n": "N",  # Supraventricular escape beat
    # Class S: Supraventricular Ectopic Beats (SVEB)
    "A": "S",  # Atrial premature contraction (APC)
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Premature or ectopic supraventricular beat
    # Class V: Ventricular Ectopic Beats (VEB)
    "V": "V",  # Premature ventricular contraction (PVC)
    "E": "V",  # Ventricular escape beat
    "r": "V",  # R-on-T premature ventricular contraction
    # Class F: Fusion of Ventricular and Normal
    "F": "F",  # Fusion beat
    # Class Q: Unknown / Paced / Unclassifiable
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
    "?": "Q",  # Beat not classified during learning phase
}

# Non-Beat Auxiliary Markers that must be filtered out per ANSI/AAMI EC57
NON_BEAT_SYMBOLS: Set[str] = {
    "[",  # Start of ventricular flutter/fibrillation
    "!",  # Ventricular flutter wave
    "]",  # End of ventricular flutter/fibrillation
    "x",  # Non-conducted P-wave (blocked APC)
    "(",  # Waveform onset
    ")",  # Waveform end
    "p",  # P-wave peak
    "t",  # T-wave peak
    "u",  # U-wave peak
    "'",  # PQ junction / J-point marker
    "^",  # Non-conducted pacemaker spike
    "|",  # Isolated QRS-like artifact
    "~",  # Signal quality change / noise epoch
    "+",  # Rhythm change marker (e.g. (AFIB, (N)
    '"',  # Comment text / note
    "@",  # Web / file link
}

# Integer class encoding
AAMI_CLASS_TO_INT: Dict[str, int] = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}

INT_TO_AAMI_CLASS: Dict[int, str] = {v: k for k, v in AAMI_CLASS_TO_INT.items()}


def map_symbol_to_aami(symbol: str) -> Optional[str]:
    """
    Map a single annotation symbol to its ANSI/AAMI EC57 superclass.

    Returns:
        One of {'N', 'S', 'V', 'F', 'Q'} if valid beat symbol,
        or None if non-beat or unmapped symbol.
    """
    if not symbol:
        return None
    sym = symbol.strip()
    if sym in AAMI_BEAT_MAP:
        return AAMI_BEAT_MAP[sym]
    return None


def is_qrs_beat(symbol: str) -> bool:
    """Return True if the symbol is a valid QRS beat annotation."""
    if not symbol:
        return False
    sym = symbol.strip()
    return sym in AAMI_BEAT_MAP and sym not in NON_BEAT_SYMBOLS


def normalize_annotations(
    annotations: Any,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract sample indices and annotation symbols from various input formats.
    Supports:
    - List of (sample_idx, symbol_str) tuples
    - wfdb.Annotation object with .sample and .symbol
    - Dict with 'sample' and 'symbol' keys
    - Tuple of (samples_array, symbols_list)
    """
    if hasattr(annotations, "sample") and hasattr(annotations, "symbol"):
        samples = np.asarray(annotations.sample, dtype=int)
        symbols = [str(s) for s in annotations.symbol]
        return samples, symbols

    if isinstance(annotations, dict) and "sample" in annotations and "symbol" in annotations:
        samples = np.asarray(annotations["sample"], dtype=int)
        symbols = [str(s) for s in annotations["symbol"]]
        return samples, symbols

    if isinstance(annotations, (tuple, list)):
        if len(annotations) == 2 and isinstance(annotations[0], (np.ndarray, list)) and isinstance(annotations[1], (np.ndarray, list)):
            samples = np.asarray(annotations[0], dtype=int)
            symbols = [str(s) for s in annotations[1]]
            return samples, symbols

        # List of (sample, symbol) tuples
        samples = []
        symbols = []
        for item in annotations:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                samples.append(int(item[0]))
                symbols.append(str(item[1]))
            elif hasattr(item, "sample") and hasattr(item, "symbol"):
                samples.append(int(item.sample))
                symbols.append(str(item.symbol))
        return np.asarray(samples, dtype=int), symbols

    raise ValueError(f"Unsupported annotation format: {type(annotations)}")


def compute_rhythm_features(
    r_peaks: np.ndarray,
    fs: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute 4-dimensional rhythm features for each beat:
    [RR_pre, RR_post, delta_RR, RR_ratio].

    Args:
        r_peaks: Sample indices of R-peaks, shape (N_beats,).
        fs: Sampling rate in Hz.
        eps: Small constant to prevent division by zero.

    Returns:
        np.ndarray of shape (N_beats, 4), dtype float32.
    """
    r_peaks_arr = np.asarray(r_peaks, dtype=np.float64)
    n_beats = len(r_peaks_arr)
    if n_beats == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if n_beats == 1:
        # Single beat fallback
        default_rr = 0.8  # ~75 bpm default
        features = np.array([[default_rr, default_rr, 0.0, 1.0]], dtype=np.float32)
        return features

    # Check monotonicity; sort internally if unsorted while tracking inverse mapping
    is_sorted = bool(np.all(r_peaks_arr[:-1] <= r_peaks_arr[1:]))
    if not is_sorted:
        sort_idx = np.argsort(r_peaks_arr, kind="stable")
        r_peaks_sorted = r_peaks_arr[sort_idx]
    else:
        sort_idx = None
        r_peaks_sorted = r_peaks_arr

    # Compute pairwise RR intervals in seconds with non-negativity clamp
    rr_intervals = np.maximum(np.diff(r_peaks_sorted) / float(fs), 0.0)

    rr_pre = np.zeros(n_beats, dtype=np.float64)
    rr_post = np.zeros(n_beats, dtype=np.float64)

    # First beat: RR_pre set to first RR_post
    rr_pre[0] = rr_intervals[0]
    rr_pre[1:] = rr_intervals

    # Last beat: RR_post set to last RR_pre
    rr_post[:-1] = rr_intervals
    rr_post[-1] = rr_intervals[-1]

    # delta_RR = RR_pre - RR_post (signed differential dynamic)
    delta_rr = rr_pre - rr_post

    # RR_ratio = RR_pre / (RR_post + eps) (strictly non-negative)
    rr_ratio = np.maximum(rr_pre / (rr_post + eps), 0.0)

    features = np.column_stack([rr_pre, rr_post, delta_rr, rr_ratio])

    # If input was unsorted, restore original caller ordering
    if sort_idx is not None:
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        features = features[inv_idx]

    return features.astype(np.float32)


def extract_heartbeats(
    record_id: str,
    raw_signal: np.ndarray,
    fs: float,
    annotations: Any,
    target_fs: float = 360.0,
    lead: int = 0,
    pre_samples: int = 120,
    post_samples: int = 240,
    allow_padding: bool = False,
    normalize_beats: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract segmented heartbeats and clinical rhythm dynamics under PROJECT.md Contract 1.

    Args:
        record_id: Unique record identifier string.
        raw_signal: 1D or 2D ECG signal array.
        fs: Original sampling frequency in Hz.
        annotations: Beat annotations (list of tuples, wfdb.Annotation, or dict).
        target_fs: Target sampling frequency (default: 360.0 Hz).
        lead: Index of lead to extract if multi-channel (default: 0).
        pre_samples: Samples before R-peak (default: 120 samples = ~333 ms at 360 Hz).
        post_samples: Samples after R-peak (default: 240 samples = ~667 ms at 360 Hz).
        allow_padding: If False (default), beats truncated by signal boundaries are omitted.
        normalize_beats: If True (default), each extracted beat is z-score normalized.

    Returns:
        Dictionary adhering strictly to PROJECT.md Contract 1:
        - 'beats': np.ndarray of shape (N_beats, 1, 360), dtype float32.
        - 'rhythm_features': np.ndarray of shape (N_beats, 4), dtype float32.
        - 'labels': np.ndarray of shape (N_beats,), integer encoded 0..4 (N, S, V, F, Q).
        - 'r_peaks': np.ndarray of shape (N_beats,), sample indices at target_fs.
    """
    window_len = pre_samples + post_samples

    # 1. Parse and extract 1D lead signal
    sig_arr = np.asarray(raw_signal)

    # Early guard for empty signal
    if sig_arr.size == 0:
        return {
            "beats": np.zeros((0, 1, window_len), dtype=np.float32),
            "rhythm_features": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=int),
            "r_peaks": np.zeros((0,), dtype=int),
        }

    if sig_arr.ndim == 2:
        # Check if shape is (N_samples, N_channels) or (N_channels, N_samples)
        if sig_arr.shape[0] >= sig_arr.shape[1]:
            # (N_samples, N_channels)
            lead_idx = min(lead, sig_arr.shape[1] - 1)
            sig_1d = sig_arr[:, lead_idx]
        else:
            # (N_channels, N_samples)
            lead_idx = min(lead, sig_arr.shape[0] - 1)
            sig_1d = sig_arr[lead_idx, :]
    elif sig_arr.ndim == 1:
        sig_1d = sig_arr
    else:
        raise ValueError(f"Signal must be 1D or 2D, got shape {sig_arr.shape}")

    if len(sig_1d) == 0:
        return {
            "beats": np.zeros((0, 1, window_len), dtype=np.float32),
            "rhythm_features": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=int),
            "r_peaks": np.zeros((0,), dtype=int),
        }

    # 2. Preprocess: Bandpass filter and resample to target_fs
    preprocessor = ECGPreprocessor(target_fs=target_fs, normalize=False)
    processed_signal = preprocessor.preprocess(sig_1d, fs=fs, target_fs=target_fs)

    # 3. Parse annotations and filter non-beats & out-of-domain samples
    raw_samples, raw_symbols = normalize_annotations(annotations)

    valid_samples = []
    valid_classes = []
    orig_len = len(sig_1d)
    for s_idx, sym in zip(raw_samples, raw_symbols):
        if is_qrs_beat(sym):
            aami_class = map_symbol_to_aami(sym)
            if aami_class is not None:
                # Retain only annotations within the recorded signal domain [0, orig_len)
                if 0 <= s_idx < orig_len:
                    valid_samples.append(s_idx)
                    valid_classes.append(AAMI_CLASS_TO_INT[aami_class])

    if len(valid_samples) == 0:
        return {
            "beats": np.zeros((0, 1, window_len), dtype=np.float32),
            "rhythm_features": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=int),
            "r_peaks": np.zeros((0,), dtype=int),
        }

    # Chronologically sort valid beats by sample index to ensure:
    # 1. Monotonically non-decreasing beat sequence
    # 2. Strict non-negativity for RR intervals (RR_pre >= 0, RR_post >= 0)
    # 3. Strict non-negativity for RR ratios (RR_ratio >= 0)
    # 4. Perfect 1:1 invariant alignment between beats, labels, rhythm_features, and r_peaks
    if len(valid_samples) > 1:
        sort_idx = np.argsort(valid_samples, kind="stable")
        valid_samples = [valid_samples[i] for i in sort_idx]
        valid_classes = [valid_classes[i] for i in sort_idx]

    # 4. Scale sample indices if fs != target_fs
    scaled_peaks = resample_indices(valid_samples, orig_fs=fs, target_fs=target_fs)

    # 5. Compute rhythm dynamics (computed across all valid consecutive R-peaks)
    rhythm_feats = compute_rhythm_features(scaled_peaks, fs=target_fs)

    # 6. Extract beat windows
    sig_len = len(processed_signal)
    beats_list = []
    labels_list = []
    feats_list = []
    peaks_list = []

    for i, (r_peak, cls_int) in enumerate(zip(scaled_peaks, valid_classes)):
        # Guard: R-peak fiducial must be inside the resampled signal domain
        if r_peak < 0 or r_peak >= sig_len:
            continue

        start = r_peak - pre_samples
        end = r_peak + post_samples

        if start < 0 or end > sig_len:
            if not allow_padding:
                # Boundary beat omitted per standard clinical protocol
                continue
            # Defensively compute bounded non-negative slice indices
            clip_start = max(0, min(sig_len, start))
            clip_end = max(0, min(sig_len, end))
            if clip_start >= clip_end:
                continue

            segment = processed_signal[clip_start:clip_end]
            if len(segment) == 0:
                continue

            pad_left = max(0, -start)
            pad_right = max(0, end - sig_len)
            segment = np.pad(segment, (pad_left, pad_right), mode="edge")
        else:
            segment = processed_signal[start:end]

        if len(segment) != window_len:
            continue

        # Per-beat z-score standardization
        if normalize_beats:
            b_mean = np.mean(segment)
            b_std = np.std(segment)
            segment = (segment - b_mean) / (b_std + 1e-8)

        beats_list.append(segment.astype(np.float32))
        labels_list.append(cls_int)
        feats_list.append(rhythm_feats[i])
        peaks_list.append(r_peak)

    if len(beats_list) == 0:
        return {
            "beats": np.zeros((0, 1, window_len), dtype=np.float32),
            "rhythm_features": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=int),
            "r_peaks": np.zeros((0,), dtype=int),
        }

    # Format arrays
    beats_arr = np.array(beats_list, dtype=np.float32)[:, np.newaxis, :]  # (N_beats, 1, 360)
    feats_arr = np.array(feats_list, dtype=np.float32)                   # (N_beats, 4)
    labels_arr = np.array(labels_list, dtype=int)                        # (N_beats,)
    peaks_arr = np.array(peaks_list, dtype=int)                          # (N_beats,)

    return {
        "beats": beats_arr,
        "rhythm_features": feats_arr,
        "labels": labels_arr,
        "r_peaks": peaks_arr,
    }
