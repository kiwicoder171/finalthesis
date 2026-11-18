"""
Lightweight WFDB Streamer and Selective Ingestion Module.

Enables selective HTTP streaming and local caching of PhysioNet ECG databases:
- MIT-BIH Arrhythmia Database (mitdb)
- PTB-XL (ptb-xl)
- CPSC 2018 (challenge-2020)
- Chapman-Shaoxing (ecg-arrhythmia)
- St Petersburg INCART (incartdb)

Avoids multi-gigabyte raw archive downloads by streaming only the requested records
and sample windows, with automatic disk caching and deterministic offline synthetic fallback.
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import numpy as np
import scipy.io
import wfdb


class WFDBStreamer:
    """Lightweight selective streamer for clinical ECG databases."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout_sec: int = 20,
    ):
        """
        Initialize streamer with optional local disk cache.

        Args:
            cache_dir: Directory where streamed records are cached locally.
                       If None, defaults to './data/benchmarks'.
            timeout_sec: Network request timeout in seconds.
        """
        if cache_dir is None:
            self.cache_dir = Path.cwd() / "data" / "benchmarks"
        else:
            self.cache_dir = Path(cache_dir)
        self.timeout_sec = int(timeout_sec)
        self._user_agent = "Mozilla/5.0 (Final-Thesis-ECG-Benchmark/1.0)"

    def _ensure_dir(self, directory: Path) -> Path:
        """Create directory if it does not exist."""
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # -------------------------------------------------------------------------
    # 1. MIT-BIH Arrhythmia Database Streamer
    # -------------------------------------------------------------------------
    def stream_mitbih(
        self,
        record_id: str,
        sampfrom: int = 0,
        sampto: Optional[int] = None,
        use_cache: bool = True,
        fallback_to_synthetic: bool = True,
    ) -> Tuple[wfdb.Record, wfdb.Annotation]:
        """
        Stream or load a single MIT-BIH record with annotations.

        Args:
            record_id: MIT-BIH record name (e.g. '100', '101', '201').
            sampfrom: Starting sample index.
            sampto: Ending sample index (None for full recording).
            use_cache: If True, checks and saves to local cache_dir/mitdb.
            fallback_to_synthetic: If True, creates a synthetic record if offline.

        Returns:
            Tuple of (wfdb.Record, wfdb.Annotation).
        """
        rec_name = str(record_id).strip()
        db_dir = self.cache_dir / "mitdb"

        # Check local cache first
        if use_cache and db_dir.exists():
            hea_file = db_dir / f"{rec_name}.hea"
            dat_file = db_dir / f"{rec_name}.dat"
            atr_file = db_dir / f"{rec_name}.atr"
            if hea_file.exists() and dat_file.exists() and atr_file.exists():
                local_rec = str(db_dir / rec_name)
                record = wfdb.rdrecord(
                    local_rec,
                    sampfrom=sampfrom,
                    sampto=sampto,
                )
                ann = wfdb.rdann(
                    local_rec,
                    "atr",
                    sampfrom=sampfrom,
                    sampto=sampto,
                )
                return record, ann

        # Stream from PhysioNet
        try:
            record = wfdb.rdrecord(
                rec_name,
                pn_dir="mitdb/1.0.0",
                sampfrom=sampfrom,
                sampto=sampto,
            )
            ann = wfdb.rdann(
                rec_name,
                "atr",
                pn_dir="mitdb/1.0.0",
                sampfrom=sampfrom,
                sampto=sampto,
            )

            # Optionally cache full record files if caching enabled
            if use_cache and sampfrom == 0 and sampto is None:
                self._ensure_dir(db_dir)
                try:
                    wfdb.wrsamp(
                        rec_name,
                        fs=record.fs,
                        units=record.units,
                        sig_name=record.sig_name,
                        p_signal=record.p_signal,
                        fmt=record.fmt,
                        write_dir=str(db_dir),
                    )
                    wfdb.wrann(
                        rec_name,
                        "atr",
                        sample=ann.sample,
                        symbol=ann.symbol,
                        write_dir=str(db_dir),
                    )
                except Exception:
                    pass  # Non-fatal caching error

            return record, ann

        except Exception as e:
            if fallback_to_synthetic:
                return self.create_synthetic_record(
                    record_name=rec_name,
                    duration_sec=10.0 if sampto is None else (sampto - sampfrom) / 360.0,
                    fs=360.0,
                    num_leads=2,
                    hr_bpm=72.0,
                )
            raise ConnectionError(f"Failed to stream MIT-BIH record {rec_name}: {e}") from e

    # -------------------------------------------------------------------------
    # 2. PTB-XL Streamer
    # -------------------------------------------------------------------------
    def stream_ptbxl(
        self,
        record_id: str,
        sampling_rate: int = 100,
        use_cache: bool = True,
        fallback_to_synthetic: bool = True,
    ) -> wfdb.Record:
        """
        Stream or load a PTB-XL 12-lead record.

        Args:
            record_id: Record id string (e.g. '00001', '00002') or full path.
            sampling_rate: 100 or 500 Hz.
            use_cache: Whether to use local cache.
            fallback_to_synthetic: Offline synthetic fallback.

        Returns:
            wfdb.Record containing 12-lead signal and metadata.
        """
        raw_id = str(record_id).strip()
        num_str = raw_id.split("/")[-1].split("_")[0].zfill(5)
        folder = num_str[:2] + "000"
        suffix = "lr" if sampling_rate == 100 else "hr"
        rec_name = f"{num_str}_{suffix}"
        res_dir = f"records{sampling_rate}"

        db_dir = self.cache_dir / "ptb-xl" / res_dir / folder

        if use_cache and db_dir.exists():
            hea = db_dir / f"{rec_name}.hea"
            dat = db_dir / f"{rec_name}.dat"
            if hea.exists() and dat.exists():
                return wfdb.rdrecord(str(db_dir / rec_name))

        pn_path = f"ptb-xl/1.0.3/{res_dir}/{folder}"
        try:
            record = wfdb.rdrecord(rec_name, pn_dir=pn_path)
            if use_cache:
                self._ensure_dir(db_dir)
                try:
                    wfdb.wrsamp(
                        rec_name,
                        fs=record.fs,
                        units=record.units,
                        sig_name=record.sig_name,
                        p_signal=record.p_signal,
                        fmt=record.fmt,
                        write_dir=str(db_dir),
                    )
                except Exception:
                    pass
            return record
        except Exception as e:
            if fallback_to_synthetic:
                rec, _ = self.create_synthetic_record(
                    record_name=rec_name,
                    duration_sec=10.0,
                    fs=float(sampling_rate),
                    num_leads=12,
                    hr_bpm=70.0,
                )
                return rec
            raise ConnectionError(f"Failed to stream PTB-XL record {rec_name}: {e}") from e

    # -------------------------------------------------------------------------
    # 3. CPSC 2018 Streamer
    # -------------------------------------------------------------------------
    def stream_cpsc2018(
        self,
        record_id: str,
        group: str = "g1",
        use_cache: bool = True,
        fallback_to_synthetic: bool = True,
    ) -> Dict[str, Any]:
        """
        Stream or load a CPSC 2018 12-lead MATLAB/WFDB record.

        Returns:
            Dict with 'signal' (12, N_samples), 'fs' (500.0), 'lead_names', 'comments'.
        """
        rec_name = str(record_id).strip()
        if not rec_name.startswith("A"):
            rec_name = f"A{int(rec_name):04d}"

        db_dir = self.cache_dir / "cpsc_2018" / group
        mat_file = db_dir / f"{rec_name}.mat"

        if use_cache and mat_file.exists():
            mat_data = scipy.io.loadmat(str(mat_file))
            val = mat_data["val"]
            return {
                "record_name": rec_name,
                "signal": val,
                "fs": 500.0,
                "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "source": "cache",
            }

        url = f"https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/{group}/{rec_name}.mat"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                mat_bytes = resp.read()

            if use_cache:
                self._ensure_dir(db_dir)
                with open(mat_file, "wb") as f:
                    f.write(mat_bytes)

            mat_data = scipy.io.loadmat(io.BytesIO(mat_bytes))
            val = mat_data["val"]
            return {
                "record_name": rec_name,
                "signal": val,
                "fs": 500.0,
                "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "source": "physionet_stream",
            }
        except Exception as e:
            if fallback_to_synthetic:
                rec, _ = self.create_synthetic_record(
                    record_name=rec_name,
                    duration_sec=15.0,
                    fs=500.0,
                    num_leads=12,
                    hr_bpm=75.0,
                )
                return {
                    "record_name": rec_name,
                    "signal": rec.p_signal.T,  # (12, N_samples)
                    "fs": 500.0,
                    "lead_names": rec.sig_name,
                    "source": "synthetic_fallback",
                }
            raise ConnectionError(f"Failed to stream CPSC 2018 record {rec_name}: {e}") from e

    # -------------------------------------------------------------------------
    # 4. Chapman-Shaoxing Streamer
    # -------------------------------------------------------------------------
    def stream_chapman(
        self,
        record_id: str = "JS00001",
        subfolder: str = "01/010",
        use_cache: bool = True,
        fallback_to_synthetic: bool = True,
    ) -> Dict[str, Any]:
        """
        Stream or load a Chapman-Shaoxing 12-lead record.
        """
        rec_name = str(record_id).strip()
        db_dir = self.cache_dir / "chapman" / subfolder
        mat_file = db_dir / f"{rec_name}.mat"

        if use_cache and mat_file.exists():
            mat_data = scipy.io.loadmat(str(mat_file))
            val = mat_data["val"]
            return {
                "record_name": rec_name,
                "signal": val,
                "fs": 500.0,
                "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "source": "cache",
            }

        url = f"https://physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/{subfolder}/{rec_name}.mat"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                mat_bytes = resp.read()

            if use_cache:
                self._ensure_dir(db_dir)
                with open(mat_file, "wb") as f:
                    f.write(mat_bytes)

            mat_data = scipy.io.loadmat(io.BytesIO(mat_bytes))
            val = mat_data["val"]
            return {
                "record_name": rec_name,
                "signal": val,
                "fs": 500.0,
                "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "source": "physionet_stream",
            }
        except Exception as e:
            if fallback_to_synthetic:
                rec, _ = self.create_synthetic_record(
                    record_name=rec_name,
                    duration_sec=10.0,
                    fs=500.0,
                    num_leads=12,
                    hr_bpm=72.0,
                )
                return {
                    "record_name": rec_name,
                    "signal": rec.p_signal.T,
                    "fs": 500.0,
                    "lead_names": rec.sig_name,
                    "source": "synthetic_fallback",
                }
            raise ConnectionError(f"Failed to stream Chapman record {rec_name}: {e}") from e

    # -------------------------------------------------------------------------
    # 5. St Petersburg INCART Streamer
    # -------------------------------------------------------------------------
    def stream_incart(
        self,
        record_id: str = "I01",
        sampfrom: int = 0,
        sampto: Optional[int] = None,
        use_cache: bool = True,
        fallback_to_synthetic: bool = True,
    ) -> Tuple[wfdb.Record, wfdb.Annotation]:
        """
        Stream or load a St Petersburg INCART 12-lead record.
        """
        rec_name = str(record_id).strip()
        db_dir = self.cache_dir / "incartdb"

        if use_cache and db_dir.exists():
            hea = db_dir / f"{rec_name}.hea"
            dat = db_dir / f"{rec_name}.dat"
            atr = db_dir / f"{rec_name}.atr"
            if hea.exists() and dat.exists() and atr.exists():
                local_rec = str(db_dir / rec_name)
                record = wfdb.rdrecord(local_rec, sampfrom=sampfrom, sampto=sampto)
                ann = wfdb.rdann(local_rec, "atr", sampfrom=sampfrom, sampto=sampto)
                return record, ann

        try:
            record = wfdb.rdrecord(rec_name, pn_dir="incartdb/1.0.0", sampfrom=sampfrom, sampto=sampto)
            ann = wfdb.rdann(rec_name, "atr", pn_dir="incartdb/1.0.0", sampfrom=sampfrom, sampto=sampto)

            # Cache full record files locally if caching is enabled
            if use_cache and sampfrom == 0 and sampto is None:
                self._ensure_dir(db_dir)
                try:
                    wfdb.wrsamp(
                        rec_name,
                        fs=record.fs,
                        units=record.units,
                        sig_name=record.sig_name,
                        p_signal=record.p_signal,
                        fmt=record.fmt,
                        adc_gain=record.adc_gain,
                        baseline=record.baseline,
                        write_dir=str(db_dir),
                    )
                    wfdb.wrann(
                        rec_name,
                        "atr",
                        sample=ann.sample,
                        symbol=ann.symbol,
                        write_dir=str(db_dir),
                    )
                except Exception:
                    pass  # Non-fatal caching error

            return record, ann
        except Exception as e:
            if fallback_to_synthetic:
                return self.create_synthetic_record(
                    record_name=rec_name,
                    duration_sec=10.0 if sampto is None else (sampto - sampfrom) / 257.0,
                    fs=257.0,
                    num_leads=12,
                    hr_bpm=75.0,
                )
            raise ConnectionError(f"Failed to stream INCART record {rec_name}: {e}") from e

    # -------------------------------------------------------------------------
    # 6. Realistic Synthetic ECG Record & Annotation Generator
    # -------------------------------------------------------------------------
    @staticmethod
    def create_synthetic_record(
        record_name: str = "synthetic",
        duration_sec: float = 10.0,
        fs: float = 360.0,
        num_leads: int = 2,
        hr_bpm: float = 75.0,
        add_arrhythmias: bool = True,
    ) -> Tuple[wfdb.Record, wfdb.Annotation]:
        """
        Generate a physiologically realistic multi-lead ECG waveform and matching
        beat annotations. Useful for deterministic offline unit testing.

        Simulates:
        - P, Q, R, S, T waves via Gaussian summation.
        - Realistic RR intervals with respiratory sinus arrhythmia (RSA).
        - Occasional ventricular premature contractions (V) and atrial premature beats (S).
        - Authentic wfdb.Record and wfdb.Annotation objects.
        """
        total_samples = int(duration_sec * fs)
        t = np.arange(total_samples) / fs

        # Beat generation
        beat_interval = 60.0 / hr_bpm  # seconds per beat
        r_peaks = []
        symbols = []

        curr_time = 0.4  # First peak offset
        beat_idx = 0

        while curr_time < duration_sec - 0.4:
            r_sample = int(round(curr_time * fs))
            r_peaks.append(r_sample)

            # Assign symbol: mostly N, occasional V or S if enabled
            if add_arrhythmias and beat_idx in (3, 7):
                sym = "V"  # PVC
            elif add_arrhythmias and beat_idx in (5,):
                sym = "A"  # APC (SVEB)
            elif add_arrhythmias and beat_idx in (9,):
                sym = "F"  # Fusion
            else:
                sym = "N"

            symbols.append(sym)

            # Next beat with minor RR jitter
            jitter = 0.05 * np.sin(2 * np.pi * curr_time / 4.0)  # RSA
            curr_time += beat_interval + jitter
            beat_idx += 1

        # Synthesize ECG waveform
        signal_data = np.zeros((total_samples, num_leads), dtype=np.float64)

        for lead_idx in range(num_leads):
            lead_factor = 1.0 if lead_idx == 0 else 0.7
            lead_sig = np.zeros(total_samples, dtype=np.float64)

            for peak_sample, sym in zip(r_peaks, symbols):
                # P-QRS-T Gaussian parameters relative to R peak
                if sym == "V":
                    # Wide, abnormal QRS for ventricular ectopic beat
                    components = [
                        (0.0, 1.8 * lead_factor, 0.045),     # Wide R peak
                        (0.06, -0.6 * lead_factor, 0.06),   # Deep wide S
                        (0.25, -0.5 * lead_factor, 0.12),   # Inverted T wave
                    ]
                else:
                    # Normal / supraventricular complex
                    components = [
                        (-0.16, 0.15 * lead_factor, 0.025), # P wave
                        (-0.04, -0.15 * lead_factor, 0.01), # Q wave
                        (0.00, 1.20 * lead_factor, 0.018),  # R peak
                        (0.04, -0.25 * lead_factor, 0.015), # S wave
                        (0.24, 0.30 * lead_factor, 0.050),  # T wave
                    ]

                for dt_sec, amp, width_sec in components:
                    center_sample = peak_sample + int(round(dt_sec * fs))
                    idx_range = int(round(4 * width_sec * fs))
                    left = max(0, center_sample - idx_range)
                    right = min(total_samples, center_sample + idx_range + 1)
                    if right > left:
                        sample_sub = np.arange(left, right)
                        gaussian = amp * np.exp(-0.5 * ((sample_sub - center_sample) / (width_sec * fs)) ** 2)
                        lead_sig[left:right] += gaussian

            # Add mild baseline drift (0.15 Hz) and white noise
            drift = 0.05 * np.sin(2 * np.pi * 0.15 * t)
            noise = np.random.RandomState(42 + lead_idx).normal(0.0, 0.015, total_samples)
            signal_data[:, lead_idx] = lead_sig + drift + noise

        # Construct wfdb.Record with clinical standard 16-bit WFDB parameters
        lead_names = (
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            if num_leads == 12
            else [f"MLII" if i == 0 else f"V{i}" for i in range(num_leads)]
        )
        record = wfdb.Record(
            record_name=record_name,
            p_signal=signal_data.astype(np.float32),
            fs=int(fs),
            sig_name=lead_names,
            units=["mV"] * num_leads,
            comments=[f"Synthetic ECG record for testing ({len(r_peaks)} beats)"],
            fmt=["16"] * num_leads,
            adc_gain=[200.0] * num_leads,
            baseline=[0] * num_leads,
        )

        # Construct wfdb.Annotation
        ann = wfdb.Annotation(
            record_name=record_name,
            extension="atr",
            sample=np.array(r_peaks, dtype=int),
            symbol=symbols,
            fs=int(fs),
        )

        return record, ann
