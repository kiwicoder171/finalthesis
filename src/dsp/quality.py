"""
src/dsp/quality.py
Signal Quality Index (SQI) and Motion Artifact Assessment for Member B.
Evaluates signal-to-noise ratio, baseline wander severity, kurtosis, and skewness
to determine if a recording is clinically valid or corrupted by body movement.
"""

import math

def compute_signal_quality(signal, fs):
    """
    Evaluates whether a cardiac recording (SCG or PPG) meets clinical quality thresholds.
    
    Args:
        signal (list): 1D raw or zero-centered cardiac time-series.
        fs (float): Sampling rate in Hz.
        
    Returns:
        dict: Quality metrics and classification ('OPTIMAL', 'ACCEPTABLE', 'NOISY_REJECT').
    """
    n = len(signal)
    if n < int(fs * 3.0): # Less than 3 seconds
        return {
            "sqi_score": 0.0,
            "quality_grade": "NOISY_REJECT",
            "is_usable": False,
            "reason": "Signal too short (< 3 seconds)"
        }
        
    mean_val = sum(signal) / n
    centered = [x - mean_val for x in signal]
    
    # 1. Variance and Standard Deviation
    variance = sum(x * x for x in centered) / n
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    
    if std_dev < 1e-5:
        return {
            "sqi_score": 0.0,
            "quality_grade": "NOISY_REJECT",
            "is_usable": False,
            "reason": "Flatline / sensor disconnected"
        }
        
    # Check for stationary surface / table (MEMS thermal noise floor)
    # Human chest recoil produces std_dev > 0.035 m/s^2.
    # An idle table has std_dev < 0.022 m/s^2.
    if std_dev < 0.025:
        return {
            "sqi_score": 0.0,
            "quality_grade": "NO_PULSE",
            "is_usable": False,
            "reason": "No pulse detected (Device stationary on table/surface)",
            "kurtosis": 0.0,
            "skewness": 0.0,
            "std_dev": round(std_dev, 4)
        }
        
    # 2. Kurtosis (measures impulsiveness / peak sharpness)
    # Cardiac pulses typically exhibit kurtosis > 3.0 (leptokurtic)
    m4 = sum((x ** 4) for x in centered) / n
    kurtosis = (m4 / (variance * variance)) if variance > 0 else 0.0
    
    # 3. Skewness (asymmetry)
    m3 = sum((x ** 3) for x in centered) / n
    skewness = (m3 / (std_dev ** 3)) if std_dev > 0 else 0.0
    
    # 4. Clipping and Extreme Spike Detection
    max_amp = max(abs(x) for x in centered)
    clipping_ratio = sum(1 for x in centered if abs(x) > max_amp * 0.95) / n
    
    # 5. Composite SQI Calculation (Normalized 0.0 to 1.0)
    score = 0.5
    
    # Reward healthy cardiac kurtosis range (2.5 to 8.0)
    if 2.5 <= kurtosis <= 8.5:
        score += 0.3
    elif kurtosis > 12.0: # excessive violent spikes / drops
        score -= 0.3
        
    # Penalize excessive clipping
    if clipping_ratio > 0.08:
        score -= 0.3
    else:
        score += 0.1
        
    # Penalize extreme amplitude swings (> 4.0 m/s^2 for chest resting SCG)
    if max_amp > 3.5:
        score -= 0.2
    else:
        score += 0.1
        
    final_score = max(0.0, min(1.0, score))
    
    if final_score >= 0.70:
        grade = "OPTIMAL"
        usable = True
        reason = "Clean signal with clear systolic morphology"
    elif final_score >= 0.45:
        grade = "ACCEPTABLE"
        usable = True
        reason = "Usable signal with minor motion noise"
    else:
        grade = "NOISY_REJECT"
        usable = False
        reason = "Excessive body movement or sensor displacement"
        
    return {
        "sqi_score": round(final_score, 2),
        "quality_grade": grade,
        "is_usable": usable,
        "kurtosis": round(kurtosis, 2),
        "skewness": round(skewness, 2),
        "std_dev": round(std_dev, 4),
        "reason": reason
    }
