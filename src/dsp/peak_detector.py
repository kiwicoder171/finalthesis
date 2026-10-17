"""
src/dsp/peak_detector.py
Adaptive systolic peak detector for cardiac pulses (SCG and PPG).
Includes moving-window energy integration, dynamic thresholding,
physiological refractory blanking, and ectopic beat outlier rejection.
"""

import math

def detect_systolic_peaks(filtered_signal, fs, min_bpm=40, max_bpm=180):
    """
    Detects systolic peaks corresponding to individual cardiac cycles.
    
    Args:
        filtered_signal (list): Bandpass-filtered cardiac signal.
        fs (float): Sampling rate in Hz.
        min_bpm (int): Lower physiological heart rate bound (default: 40 bpm).
        max_bpm (int): Upper physiological heart rate bound (default: 180 bpm).
        
    Returns:
        list: Indices of detected systolic peaks in the signal array.
    """
    n_samples = len(filtered_signal)
    if n_samples < 10:
        return []
        
    # Minimum blanking interval between consecutive beats (e.g. 180 bpm -> 333 ms)
    min_distance = int(fs * (60.0 / max_bpm))
    
    # 1. Moving energy envelope (150 ms window)
    win_len = max(3, int(fs * 0.15))
    squared = [x * x for x in filtered_signal]
    
    envelope = [0.0] * n_samples
    half_win = win_len // 2
    for i in range(n_samples):
        start = max(0, i - half_win)
        end = min(n_samples, i + half_win + 1)
        envelope[i] = sum(squared[start:end]) / (end - start)
        
    # 2. Adaptive threshold based on overall signal energy
    mean_energy = sum(envelope) / len(envelope) if envelope else 0.0
    threshold = mean_energy * 1.2
    
    # 3. Peak localization
    peaks = []
    last_peak_idx = -min_distance
    
    for i in range(1, n_samples - 1):
        # Local maximum condition
        if filtered_signal[i] > filtered_signal[i-1] and filtered_signal[i] > filtered_signal[i+1]:
            # Energy threshold and refractory blanking period
            if envelope[i] > threshold and (i - last_peak_idx) >= min_distance:
                peaks.append(i)
                last_peak_idx = i
                
    return peaks

def compute_peak_intervals(peak_indices, timestamps, min_ms=333.0, max_ms=1500.0):
    """
    Calculates the Peak-to-Peak Interval (PPI) series from detected peak timestamps,
    applying physiological outlier rejection to filter ectopic beats.
    
    Args:
        peak_indices (list): Array of sample indices where peaks occurred.
        timestamps (list): Corresponding timestamp array in seconds.
        min_ms (float): Minimum valid beat interval in ms (default: 333 ms -> 180 bpm).
        max_ms (float): Maximum valid beat interval in ms (default: 1500 ms -> 40 bpm).
        
    Returns:
        tuple: (raw_ppi_ms, clean_ppi_ms, clean_peak_times)
    """
    if len(peak_indices) < 2:
        return [], [], []
        
    peak_times = [timestamps[p] for p in peak_indices]
    raw_ppi = []
    clean_ppi = []
    clean_times = [peak_times[0]]
    
    for i in range(1, len(peak_times)):
        interval_ms = (peak_times[i] - peak_times[i-1]) * 1000.0
        raw_ppi.append(interval_ms)
        
        # Physiological outlier check
        if min_ms <= interval_ms <= max_ms:
            clean_ppi.append(interval_ms)
            clean_times.append(peak_times[i])
            
    return raw_ppi, clean_ppi, clean_times
