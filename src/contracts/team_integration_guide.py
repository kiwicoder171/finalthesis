"""
src/contracts/team_integration_guide.py
Developer Integration Contracts & API Examples for All 4 Members.
Demonstrates how Member A, Member C, and Member D connect to Member B's engine.
"""

# ==============================================================================
# CONTRACT 1: MEMBER A (ACQUISITION) -> MEMBER B (DSP)
# ==============================================================================
"""
Member A sends batches of 3-axis accelerometer readings over WebSockets or HTTP POST.

HTTP POST Endpoint: http://<SERVER_IP>:8000/api/sensor_stream
Payload JSON Schema:
{
    "samples": [
        {"t": 0.00, "x": 0.28, "y": -3.15, "z": 9.34},
        {"t": 0.01, "x": 0.27, "y": -3.18, "z": 9.36},
        ...
    ]
}

Frequency of transmission: Every 200 ms (20 samples @ 100 Hz).
Axes required: 't' (seconds elapsed), 'z' (dorso-ventral sternal acceleration in m/s^2).
"""

# ==============================================================================
# CONTRACT 2: MEMBER B (DSP) -> MEMBER C (MACHINE LEARNING)
# ==============================================================================
def member_c_quickstart_example():
    """
    How Member C consumes Member B's output in Python to train models or predict risk.
    """
    from src.dsp import process_cardiac_recording
    
    # Example: Member C has a time array and accelerometer signal
    # (Or loads from data/training_features_corpus.csv)
    dummy_timestamps = [i * 0.01 for i in range(1000)] # 10s @ 100Hz
    dummy_az = [9.34 + 0.1 * (i % 10) for i in range(1000)]
    
    # 1. Run Member B's pipeline
    result = process_cardiac_recording(dummy_timestamps, dummy_az, mode="scg")
    
    # 2. Extract feature dictionary
    features = result["feature_vector"]
    
    # Features dictionary schema:
    # {
    #     'mean_hr_bpm': float,     # Heart Rate (40 - 180 bpm)
    #     'mean_ppi_ms': float,     # Average beat-to-beat interval (ms)
    #     'sdnn_ms': float,         # Overall autonomic variability (ms)
    #     'rmssd_ms': float,        # Vagal/parasympathetic tone (ms)
    #     'pnn50_pct': float,       # % successive interval diffs > 50ms
    #     'vlf_power': float,       # Very Low Frequency energy (0.0033 - 0.04 Hz)
    #     'lf_power': float,        # Low Frequency energy (0.04 - 0.15 Hz)
    #     'hf_power': float,        # High Frequency energy (0.15 - 0.40 Hz)
    #     'lf_hf_ratio': float,     # Sympathovagal stress index (Normal: 1.0 - 2.0; Stress: > 2.5)
    #     'lf_nu': float,           # Normalized LF (%)
    #     'hf_nu': float,           # Normalized HF (%)
    #     'sd1': float,             # Poincaré short-term dispersion
    #     'sd2': float,             # Poincaré long-term dispersion
    #     'sd_ratio': float         # SD1 / SD2 ratio
    # }
    return features


# ==============================================================================
# CONTRACT 3: MEMBER B (DSP) & MEMBER C (ML) -> MEMBER D (DYNAMICS)
# ==============================================================================
def member_d_quickstart_example(features, risk_score):
    """
    How Member D maps Member B's heart rate and Member C's risk score
    into the Van der Pol cardiac pacemaker equations.
    """
    # 1. Base intrinsic pacemaker frequency omega_0 from measured HR
    measured_bpm = features.get("mean_hr_bpm", 75.0)
    omega_0 = 2.0 * 3.14159 * (measured_bpm / 60.0)
    
    # 2. Damping parameter mu modulated by Member C's risk score (0.0 to 1.0)
    # In healthy state: mu = 1.0 (smooth stable relaxation limit cycle)
    # Under high stress/risk: mu > 2.5 (strong nonlinear relaxation, susceptible to chaos)
    mu = 1.0 + 2.0 * risk_score
    
    # 3. Van der Pol ODE state formulation:
    # d(x)/dt = y
    # d(y)/dt = mu * (1 - x^2) * y - omega_0^2 * x + F_stress(t)
    return {"omega_0": omega_0, "mu": mu}

if __name__ == "__main__":
    feat = member_c_quickstart_example()
    print("Member C features ready:", feat.keys())
