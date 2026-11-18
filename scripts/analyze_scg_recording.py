#!/usr/bin/env python3
"""
analyze_scg_recording.py
Pure-Python DSP processor for the Phyphox chest accelerometer recording.
No external dependencies required (uses standard library: math, cmath, csv, json).
"""

import csv
import math
import cmath
import os
import json

def load_data(csv_path):
    timestamps = []
    acc_x = []
    acc_y = []
    acc_z = []
    acc_abs = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 5:
                continue
            try:
                t = float(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                a = float(row[4])
                timestamps.append(t)
                acc_x.append(x)
                acc_y.append(y)
                acc_z.append(z)
                acc_abs.append(a)
            except ValueError:
                continue
                
    return timestamps, acc_x, acc_y, acc_z, acc_abs

def compute_sampling_rate(timestamps):
    diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    mean_dt = sum(diffs) / len(diffs)
    fs = 1.0 / mean_dt
    return fs, mean_dt

def butterworth_bandpass_coeffs(lowcut, highcut, fs, order=2):
    """
    Design a digital 2nd-order bandpass filter using bilinear transform.
    Returns (b, a) coefficient lists.
    """
    # Pre-warp frequencies
    omega_l = 2.0 * fs * math.tan(math.pi * lowcut / fs)
    omega_h = 2.0 * fs * math.tan(math.pi * highcut / fs)
    
    Bw = omega_h - omega_l
    omega_0_sq = omega_h * omega_l
    
    # 2nd order analog prototype bandpass transformed to digital via bilinear
    # H_a(s) = (Bw * s) / (s^2 + Bw*s + omega_0^2)
    # Substitute s = 2*fs * (1 - z^-1) / (1 + z^-1)
    C = 2.0 * fs
    C_sq = C * C
    
    # Denominator polynomial: a0 + a1*z^-1 + a2*z^-2
    a0 = C_sq + Bw * C + omega_0_sq
    a1 = 2.0 * (omega_0_sq - C_sq)
    a2 = C_sq - Bw * C + omega_0_sq
    
    # Numerator polynomial: b0 + b1*z^-1 + b2*z^-2
    b0 = Bw * C
    b1 = 0.0
    b2 = -Bw * C
    
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    return b, a

def apply_iir_filter(b, a, data):
    """
    Apply IIR filter forward and backward (zero-phase filtering).
    """
    def filter_forward(b, a, signal):
        y = [0.0] * len(signal)
        for n in range(len(signal)):
            val = b[0] * signal[n]
            if len(b) > 1 and n >= 1:
                val += b[1] * signal[n-1]
            if len(b) > 2 and n >= 2:
                val += b[2] * signal[n-2]
            if len(a) > 1 and n >= 1:
                val -= a[1] * y[n-1]
            if len(a) > 2 and n >= 2:
                val -= a[2] * y[n-2]
            y[n] = val
        return y

    # Forward pass
    y_fwd = filter_forward(b, a, data)
    # Reverse pass for zero phase shift
    y_rev = filter_forward(b, a, y_fwd[::-1])
    return y_rev[::-1]

def fft_power_spectrum(signal, fs):
    """
    Compute FFT and Power Spectral Density using Cooley-Tukey FFT.
    """
    # Zero-pad to next power of 2
    N = len(signal)
    M = 1 << (N - 1).bit_length()
    padded = signal + [0.0] * (M - N)
    
    # In-place Cooley-Tukey
    def fft(x):
        n = len(x)
        if n <= 1:
            return x
        even = fft(x[0::2])
        odd = fft(x[1::2])
        T = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
        return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

    complex_spec = fft([complex(s, 0) for s in padded])
    freqs = [k * fs / M for k in range(M // 2)]
    psd = [(abs(complex_spec[k]) ** 2) / M for k in range(M // 2)]
    return freqs, psd

def detect_systolic_peaks(filtered_signal, fs, min_bpm=40, max_bpm=180):
    """
    Detect peaks with physiological refractory blanking.
    """
    min_distance = int(fs * (60.0 / max_bpm)) # min samples between beats (~0.33s)
    
    # Moving average envelope
    win_size = int(fs * 0.15) # 150ms window
    envelope = []
    squared = [x * x for x in filtered_signal]
    
    for i in range(len(squared)):
        start = max(0, i - win_size // 2)
        end = min(len(squared), i + win_size // 2 + 1)
        envelope.append(sum(squared[start:end]) / (end - start))
        
    # Adaptive threshold: 1.5 * moving mean
    threshold = sum(envelope) / len(envelope) * 1.2
    
    peaks = []
    last_peak = -min_distance
    
    for i in range(1, len(filtered_signal) - 1):
        if filtered_signal[i] > filtered_signal[i-1] and filtered_signal[i] > filtered_signal[i+1]:
            if envelope[i] > threshold and (i - last_peak) >= min_distance:
                peaks.append(i)
                last_peak = i
                
    return peaks

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "raw_phyphox_test", "Raw Data.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print(f"Loading {csv_path}...")
    t, ax, ay, az, a_abs = load_data(csv_path)
    fs, dt = compute_sampling_rate(t)
    duration = t[-1] - t[0]
    
    print("=" * 60)
    print("RECORDING METRICS")
    print(f"Total Samples:      {len(t)}")
    print(f"Total Duration:     {duration:.2f} seconds")
    print(f"Sampling Frequency: {fs:.2f} Hz (dt = {dt*1000:.2f} ms)")
    print(f"Mean Z-axis Accel:  {sum(az)/len(az):.3f} m/s^2 (Gravity baseline)")
    print(f"Std Z-axis Accel:   {math.sqrt(sum((x - sum(az)/len(az))**2 for x in az)/len(az)):.4f} m/s^2")
    print("=" * 60)
    
    # Remove DC gravity offset
    mean_z = sum(az) / len(az)
    az_ac = [x - mean_z for x in az]
    
    # Test bandpass filtering
    # 1. Cardiac SCG Band: 1.5 Hz to 15.0 Hz
    b_cardiac, a_cardiac = butterworth_bandpass_coeffs(1.5, 15.0, fs)
    filtered_cardiac = apply_iir_filter(b_cardiac, a_cardiac, az_ac)
    
    # 2. Respiration Band: 0.15 Hz to 0.5 Hz (Chest Breathing)
    b_resp, a_resp = butterworth_bandpass_coeffs(0.15, 0.5, fs)
    filtered_resp = apply_iir_filter(b_resp, a_resp, az_ac)
    
    # Compute FFT on the cardiac signal
    freqs, psd = fft_power_spectrum(filtered_cardiac, fs)
    
    # Find dominant frequency in cardiac band (0.7 Hz to 3.0 Hz -> 42 to 180 bpm)
    cardiac_peaks = []
    for f, p in zip(freqs, psd):
        if 0.8 <= f <= 2.8:
            cardiac_peaks.append((p, f))
    cardiac_peaks.sort(reverse=True)
    
    top_freq = cardiac_peaks[0][1] if cardiac_peaks else 0
    estimated_hr = top_freq * 60.0
    
    # Detect individual systolic peaks in time domain
    peaks = detect_systolic_peaks(filtered_cardiac, fs)
    peak_times = [t[p] for p in peaks]
    
    # Calculate PPI (Peak-to-Peak Intervals)
    ppi = [(peak_times[i] - peak_times[i-1]) * 1000.0 for i in range(1, len(peak_times))]
    # Filter reasonable PPIs (350ms to 1500ms)
    clean_ppi = [p for p in ppi if 350.0 <= p <= 1500.0]
    
    mean_hr_ppi = (60000.0 / (sum(clean_ppi) / len(clean_ppi))) if clean_ppi else 0
    sdnn = math.sqrt(sum((x - (sum(clean_ppi)/len(clean_ppi)))**2 for x in clean_ppi)/(len(clean_ppi)-1)) if len(clean_ppi) > 1 else 0
    rmssd = math.sqrt(sum((clean_ppi[i] - clean_ppi[i-1])**2 for i in range(1, len(clean_ppi))) / (len(clean_ppi)-1)) if len(clean_ppi) > 1 else 0
    
    print("\n" + "=" * 60)
    print("PHYSIOLOGICAL CARDIAC EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Dominant Spectral Pulse Frequency:  {top_freq:.2f} Hz")
    print(f"Estimated Heart Rate (Spectral):     {estimated_hr:.1f} BPM")
    print(f"Total Detected Cardiac Beats:       {len(peaks)} beats")
    print(f"Mean Heart Rate (Time-Domain PPI):  {mean_hr_ppi:.1f} BPM")
    if clean_ppi:
        print(f"Mean Beat-to-Beat Interval (PPI):   {sum(clean_ppi)/len(clean_ppi):.1f} ms")
        print(f"SDNN (Overall Variability):         {sdnn:.1f} ms")
        print(f"RMSSD (Vagal Autonomic Tone):       {rmssd:.1f} ms")
    print("=" * 60)
    
    # Save a JSON report for visualization
    results = {
        "duration": duration,
        "fs": fs,
        "spectral_hr": estimated_hr,
        "ppi_hr": mean_hr_ppi,
        "beat_count": len(peaks),
        "sdnn": sdnn,
        "rmssd": rmssd,
        "timestamps": t[:800], # First 8 seconds for web visualization
        "raw_az": az[:800],
        "filtered_cardiac": filtered_cardiac[:800],
        "filtered_resp": filtered_resp[:800],
        "peaks_first_8s": [p for p in peaks if p < 800],
        "freqs": freqs[:100], # Up to 5 Hz
        "psd": psd[:100]
    }
    
    out_json = os.path.join(project_root, "data", "raw_phyphox_test", "analysis_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f)
        
    print(f"Analysis results saved to {out_json}")

if __name__ == "__main__":
    main()
