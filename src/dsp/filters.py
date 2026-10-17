"""
src/dsp/filters.py
Digital Butterworth IIR filter implementations using bilinear transform.
Written in pure Python with zero external dependencies.
Supports zero-phase (forward-backward) filtering to eliminate phase distortion.
"""

import math

def design_butterworth_bandpass(lowcut, highcut, fs):
    """
    Computes 2nd-order Butterworth bandpass filter coefficients (b, a)
    using analog prototype transformation and the bilinear transform.
    
    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        
    Returns:
        tuple: (b, a) where b and a are lists of filter coefficients.
    """
    if lowcut <= 0 or highcut >= fs / 2:
        raise ValueError(f"Cutoff frequencies must be within (0, {fs/2}) Hz")
        
    # Pre-warp frequencies into continuous-time domain
    omega_l = 2.0 * fs * math.tan(math.pi * lowcut / fs)
    omega_h = 2.0 * fs * math.tan(math.pi * highcut / fs)
    
    bw = omega_h - omega_l
    omega_0_sq = omega_h * omega_l
    
    # Bilinear transform constant C = 2 * fs
    c = 2.0 * fs
    c_sq = c * c
    
    # Denominator polynomial: a0 + a1*z^-1 + a2*z^-2
    a0 = c_sq + bw * c + omega_0_sq
    a1 = 2.0 * (omega_0_sq - c_sq)
    a2 = c_sq - bw * c + omega_0_sq
    
    # Numerator polynomial: b0 + b1*z^-1 + b2*z^-2
    b0 = bw * c
    b1 = 0.0
    b2 = -bw * c
    
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    return b, a

def apply_zero_phase_filter(b, a, signal):
    """
    Applies IIR filter forward and then backward (filtfilt equivalent)
    to achieve precisely zero phase distortion.
    
    Args:
        b (list): Numerator coefficients.
        a (list): Denominator coefficients.
        signal (list): 1D input time-series array.
        
    Returns:
        list: Filtered 1D array.
    """
    n_samples = len(signal)
    if n_samples < 4:
        return list(signal)
        
    def _filter_1d(b_coeffs, a_coeffs, data):
        y = [0.0] * len(data)
        for n in range(len(data)):
            val = b_coeffs[0] * data[n]
            if len(b_coeffs) > 1 and n >= 1:
                val += b_coeffs[1] * data[n-1]
            if len(b_coeffs) > 2 and n >= 2:
                val += b_coeffs[2] * data[n-2]
            if len(a_coeffs) > 1 and n >= 1:
                val -= a_coeffs[1] * y[n-1]
            if len(a_coeffs) > 2 and n >= 2:
                val -= a_coeffs[2] * y[n-2]
            y[n] = val
        return y

    # Forward pass
    forward_filtered = _filter_1d(b, a, signal)
    # Reverse pass (flips data, filters, and flips back)
    backward_filtered = _filter_1d(b, a, forward_filtered[::-1])
    return backward_filtered[::-1]

def filter_scg_cardiac(signal, fs):
    """
    Bandpass filters Seismocardiography (SCG) signal (1.5 Hz to 15.0 Hz).
    Isolates aortic valve opening and rapid ventricular ejection recoil.
    """
    b, a = design_butterworth_bandpass(1.5, 15.0, fs)
    return apply_zero_phase_filter(b, a, signal)

def filter_scg_respiration(signal, fs):
    """
    Bandpass filters thoracic respiratory movement (0.15 Hz to 0.50 Hz).
    Isolates slow chest expansion and breath-hold phases.
    """
    b, a = design_butterworth_bandpass(0.15, 0.50, fs)
    return apply_zero_phase_filter(b, a, signal)

def filter_ppg_cardiac(signal, fs):
    """
    Bandpass filters Photoplethysmography (PPG) optical pulse (0.5 Hz to 5.0 Hz).
    Matches 30 bpm to 300 bpm cardiac rates for camera fingertip fallback.
    """
    b, a = design_butterworth_bandpass(0.5, 5.0, fs)
    return apply_zero_phase_filter(b, a, signal)
