"""
src/dsp/synthetic_data.py
Physiological signal generator for cardiac testing and benchmarking.
Generates synthetic SCG and PPG waveforms with controllable heart rate,
autonomic variability (SDNN), sympathetic stress tone (LF/HF), and arrhythmias.
"""

import math
import random

def generate_synthetic_scg(duration_sec=30.0, fs=100.0, base_hr=75.0, stress_level="normal", seed=42):
    """
    Generates synthetic 3-axis Seismocardiogram (SCG) accelerometer time-series.
    
    Args:
        duration_sec (float): Duration in seconds.
        fs (float): Sampling frequency in Hz.
        base_hr (float): Mean heart rate in bpm.
        stress_level (str): 'normal', 'stressed', or 'arrhythmic'.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (timestamps, acc_z, label_info)
    """
    random.seed(seed)
    n_samples = int(duration_sec * fs)
    dt = 1.0 / fs
    timestamps = [i * dt for i in range(n_samples)]
    
    # 1. Base parameters by stress condition
    if stress_level == "stressed":
        hr = base_hr + 25.0 # Tachycardia
        hrv_std = 0.02 # Suppressed variability (vagal withdrawal)
        resp_freq = 0.35 # Rapid breathing (21 breaths/min)
        label = {"state": "STRESSED", "stress_class": 1, "arrhythmia": False}
    elif stress_level == "arrhythmic":
        hr = base_hr + 10.0
        hrv_std = 0.12 # High chaotic variability / ectopic beats
        resp_freq = 0.22
        label = {"state": "ARRHYTHMIC", "stress_class": 2, "arrhythmia": True}
    else: # normal
        hr = base_hr
        hrv_std = 0.06 # Healthy vagal tone
        resp_freq = 0.20 # Normal relaxed breathing (12 breaths/min)
        label = {"state": "NORMAL", "stress_class": 0, "arrhythmia": False}
        
    # 2. Generate beat timestamps using integrate-and-fire pacemaker with respiratory sinus arrhythmia (RSA)
    beat_times = []
    t_curr = 0.4
    while t_curr < duration_sec:
        # Respiratory Sinus Arrhythmia modulation
        rsa_mod = 0.08 * math.sin(2.0 * math.pi * resp_freq * t_curr)
        # Random interval perturbation
        interval = (60.0 / hr) * (1.0 + rsa_mod + random.gauss(0, hrv_std))
        
        # Insert occasional ectopic beats if arrhythmic
        if stress_level == "arrhythmic" and random.random() < 0.08:
            interval *= 0.60 # Premature ventricular contraction
            
        t_curr += max(0.35, interval)
        if t_curr < duration_sec:
            beat_times.append(t_curr)
            
    # 3. Synthesize SCG Dorso-Ventral Waveform (a_z)
    # Baseline DC gravity (~9.8 m/s^2) + slow respiration (~0.15 m/s^2)
    gravity_dc = 9.35
    signal = []
    
    for t in timestamps:
        # Slow thoracic breathing motion
        resp_motion = 0.12 * math.sin(2.0 * math.pi * resp_freq * t)
        
        # Cardiac micro-vibrations (Aortic opening AO, Rapid ejection RE, Aortic closure AC)
        cardiac_pulse = 0.0
        for b_time in beat_times:
            dt_beat = t - b_time
            if 0.0 <= dt_beat <= 0.35: # Waveform duration ~350ms
                # Damped oscillation modeling myocardial recoil
                decay = math.exp(-dt_beat / 0.06)
                ao_wave = math.sin(2.0 * math.pi * 12.0 * dt_beat) # 12 Hz mechanical resonance
                re_wave = 0.5 * math.sin(2.0 * math.pi * 6.0 * dt_beat) # 6 Hz ejection
                cardiac_pulse += 0.28 * decay * (ao_wave + re_wave)
                
        # Subtle sensor white noise
        sensor_noise = random.gauss(0, 0.015)
        
        total_az = gravity_dc + resp_motion + cardiac_pulse + sensor_noise
        signal.append(total_az)
        
    return timestamps, signal, label
