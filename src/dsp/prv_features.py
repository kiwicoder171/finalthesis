"""
src/dsp/prv_features.py
Clinical Pulse Rate Variability (PRV) feature extraction engine.
Computes time-domain, frequency-domain (FFT spectral), and non-linear metrics
to feed Member C's machine learning risk and stress classifiers.
"""

import math
import cmath

def extract_time_domain_features(clean_ppi_ms):
    """
    Computes standard statistical time-domain PRV/HRV metrics.
    
    Args:
        clean_ppi_ms (list): Cleaned beat-to-beat interval series in milliseconds.
        
    Returns:
        dict: Standard metrics {mean_hr_bpm, mean_ppi_ms, sdnn_ms, rmssd_ms, pnn50_pct}
    """
    if not clean_ppi_ms:
        return {
            "mean_hr_bpm": 0.0,
            "mean_ppi_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50_pct": 0.0
        }
        
    n = len(clean_ppi_ms)
    mean_ppi = sum(clean_ppi_ms) / n
    mean_hr = 60000.0 / mean_ppi if mean_ppi > 0 else 0.0
    
    # SDNN: Standard deviation of NN/PPI intervals
    if n > 1:
        variance = sum((x - mean_ppi) ** 2 for x in clean_ppi_ms) / (n - 1)
        sdnn = math.sqrt(variance)
    else:
        sdnn = 0.0
        
    # RMSSD: Root mean square of successive differences
    # pNN50: Percentage of successive intervals differing by > 50 ms
    successive_diffs = [clean_ppi_ms[i+1] - clean_ppi_ms[i] for i in range(n - 1)]
    if successive_diffs:
        sum_sq_diffs = sum(d * d for d in successive_diffs)
        rmssd = math.sqrt(sum_sq_diffs / len(successive_diffs))
        
        nn50_count = sum(1 for d in successive_diffs if abs(d) > 50.0)
        pnn50 = (nn50_count / len(successive_diffs)) * 100.0
    else:
        rmssd = 0.0
        pnn50 = 0.0
        
    return {
        "mean_hr_bpm": round(mean_hr, 1),
        "mean_ppi_ms": round(mean_ppi, 1),
        "sdnn_ms": round(sdnn, 1),
        "rmssd_ms": round(rmssd, 1),
        "pnn50_pct": round(pnn50, 1)
    }

def compute_fft_power_spectrum(signal, fs):
    """
    Computes discrete Power Spectral Density using Cooley-Tukey FFT.
    
    Args:
        signal (list): 1D numerical signal.
        fs (float): Sampling frequency in Hz.
        
    Returns:
        tuple: (freqs, psd)
    """
    n = len(signal)
    if n < 2:
        return [], []
        
    # Zero-pad to nearest power of 2 for FFT efficiency
    m = 1 << (n - 1).bit_length()
    padded = signal + [0.0] * (m - n)
    
    def _cooley_tukey_fft(x):
        length = len(x)
        if length <= 1:
            return x
        even = _cooley_tukey_fft(x[0::2])
        odd = _cooley_tukey_fft(x[1::2])
        factors = [cmath.exp(-2j * math.pi * k / length) * odd[k] for k in range(length // 2)]
        return [even[k] + factors[k] for k in range(length // 2)] + [even[k] - factors[k] for k in range(length // 2)]

    complex_spectrum = _cooley_tukey_fft([complex(val, 0) for val in padded])
    half_m = m // 2
    
    freqs = [k * fs / m for k in range(half_m)]
    psd = [(abs(complex_spectrum[k]) ** 2) / m for k in range(half_m)]
    return freqs, psd

def extract_frequency_domain_features(clean_ppi_ms):
    """
    Computes clinical frequency-domain PRV spectral metrics:
    - VLF Power (0.0033 to 0.04 Hz)
    - LF Power (0.04 to 0.15 Hz): Sympathetic & baroreflex activity
    - HF Power (0.15 to 0.40 Hz): Parasympathetic vagal tone
    - LF/HF Ratio: Sympathovagal autonomic balance index
    """
    if len(clean_ppi_ms) < 8:
        return {
            "vlf_power": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 1.0,
            "lf_nu": 50.0,
            "hf_nu": 50.0
        }
        
    # Resample uneven PPI intervals to uniform 4.0 Hz grid via linear interpolation
    fs_resample = 4.0
    total_time_ms = sum(clean_ppi_ms)
    total_time_s = total_time_ms / 1000.0
    
    cumulative_times = [0.0]
    for interval in clean_ppi_ms:
        cumulative_times.append(cumulative_times[-1] + (interval / 1000.0))
        
    n_resampled = int(total_time_s * fs_resample)
    if n_resampled < 8:
        return {"vlf_power": 0.0, "lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 1.0, "lf_nu": 50.0, "hf_nu": 50.0}
        
    resampled_ppi = []
    for k in range(n_resampled):
        t_target = k / fs_resample
        # Find segment in cumulative_times
        idx = 0
        while idx < len(cumulative_times) - 2 and cumulative_times[idx+1] < t_target:
            idx += 1
        t0 = cumulative_times[idx]
        t1 = cumulative_times[idx+1]
        v0 = clean_ppi_ms[min(idx, len(clean_ppi_ms)-1)]
        v1 = clean_ppi_ms[min(idx+1, len(clean_ppi_ms)-1)]
        
        # Linear interpolate
        frac = (t_target - t0) / (t1 - t0) if t1 > t0 else 0.0
        interpolated_val = v0 + frac * (v1 - v0)
        resampled_ppi.append(interpolated_val)
        
    # Remove mean (DC offset) before FFT
    mean_val = sum(resampled_ppi) / len(resampled_ppi)
    detrended = [x - mean_val for x in resampled_ppi]
    
    freqs, psd = compute_fft_power_spectrum(detrended, fs_resample)
    
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    vlf_power = 0.0
    lf_power = 0.0
    hf_power = 0.0
    
    for f, p in zip(freqs, psd):
        if 0.0033 <= f < 0.04:
            vlf_power += p * df
        elif 0.04 <= f < 0.15:
            lf_power += p * df
        elif 0.15 <= f <= 0.40:
            hf_power += p * df
            
    total_power = vlf_power + lf_power + hf_power
    lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 1.0
    
    # Normalized units (LFnu, HFnu)
    lf_hf_sum = lf_power + hf_power
    lf_nu = (lf_power / lf_hf_sum * 100.0) if lf_hf_sum > 0 else 50.0
    hf_nu = (hf_power / lf_hf_sum * 100.0) if lf_hf_sum > 0 else 50.0
    
    return {
        "vlf_power": round(vlf_power, 2),
        "lf_power": round(lf_power, 2),
        "hf_power": round(hf_power, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 3),
        "lf_nu": round(lf_nu, 1),
        "hf_nu": round(hf_nu, 1)
    }

def extract_poincare_descriptors(clean_ppi_ms):
    """
    Computes non-linear Poincaré plot dispersion metrics (SD1 and SD2).
    SD1 represents short-term variability (parasympathetic).
    SD2 represents long-term variability (sympathetic).
    """
    if len(clean_ppi_ms) < 3:
        return {"sd1": 0.0, "sd2": 0.0, "sd_ratio": 1.0}
        
    diffs = [clean_ppi_ms[i+1] - clean_ppi_ms[i] for i in range(len(clean_ppi_ms) - 1)]
    sums = [clean_ppi_ms[i+1] + clean_ppi_ms[i] for i in range(len(clean_ppi_ms) - 1)]
    
    var_diff = sum(d * d for d in diffs) / len(diffs)
    mean_sum = sum(sums) / len(sums)
    var_sum = sum((s - mean_sum) ** 2 for s in sums) / len(sums)
    
    sd1 = math.sqrt(0.5 * var_diff)
    sd2 = math.sqrt(max(0.0, 2.0 * var_sum - 0.5 * var_diff))
    sd_ratio = (sd1 / sd2) if sd2 > 0 else 1.0
    
    return {
        "sd1": round(sd1, 1),
        "sd2": round(sd2, 1),
        "sd_ratio": round(sd_ratio, 3)
    }
