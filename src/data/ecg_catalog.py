"""
ECG Database Cataloging and Registry Module.

Provides comprehensive cataloging of workspace and external clinical ECG databases:
- PhysioNet MIT-BIH Arrhythmia Database (mitdb)
- PTB-XL Large 12-lead Electrocardiography Database (ptb-xl)
- CPSC 2018 (China Physiological Signal Challenge 2018)
- Chapman-Shaoxing 12-Lead ECG Database (ecg-arrhythmia)
- St Petersburg INCART 12-Lead Arrhythmia Database (incartdb)
- WESAD (Wearable Stress and Affect Detection)
- Local Workspace Corpora (SCG feature corpus, smartphone accelerometer data)

Includes the de Chazal et al. (2004) inter-patient split definition (DS1, DS2, Paced).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata container for ECG and physiological datasets."""
    name: str
    physionet_slug: Optional[str]
    description: str
    sampling_rate_hz: float
    num_leads: int
    lead_names: List[str]
    num_records: int
    annotation_type: str
    source_url: str
    is_external: bool
    typical_duration_sec: float
    classes_supported: List[str]
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


# de Chazal et al. (2004) Inter-Patient Partitions for MIT-BIH
DS1_RECORDS: List[str] = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

DS2_RECORDS: List[str] = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

PACED_RECORDS: List[str] = ["102", "104", "107", "217"]

ALL_MITBIH_RECORDS: List[str] = sorted(DS1_RECORDS + DS2_RECORDS + PACED_RECORDS)


class ECGCatalog:
    """Central registry and discovery engine for ECG databases."""

    _REGISTRY: Dict[str, DatasetMetadata] = {
        "mitdb": DatasetMetadata(
            name="MIT-BIH Arrhythmia Database",
            physionet_slug="mitdb/1.0.0",
            description="Gold-standard ambulatory 2-channel ECG recordings with beat-by-beat annotations.",
            sampling_rate_hz=360.0,
            num_leads=2,
            lead_names=["MLII", "V1/V2/V5"],
            num_records=48,
            annotation_type="atr (QRS beat-by-beat AAMI EC57)",
            source_url="https://physionet.org/content/mitdb/1.0.0/",
            is_external=True,
            typical_duration_sec=1800.0,
            classes_supported=["N", "S", "V", "F", "Q"],
            extra_attributes={
                "evaluated_records": 44,
                "ds1_records": 22,
                "ds2_records": 22,
                "paced_records": 4,
            },
        ),
        "ptb-xl": DatasetMetadata(
            name="PTB-XL Electrocardiography Database",
            physionet_slug="ptb-xl/1.0.3",
            description="Large 12-lead clinical ECG database of 21,837 recordings from 18,885 patients.",
            sampling_rate_hz=100.0,  # Also available at 500.0 Hz
            num_leads=12,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            num_records=21837,
            annotation_type="SCP-ECG diagnostic statements and rhythm statements",
            source_url="https://physionet.org/content/ptb-xl/1.0.3/",
            is_external=True,
            typical_duration_sec=10.0,
            classes_supported=["NORM", "MI", "STTC", "CD", "HYP"],
            extra_attributes={"stratified_folds": 10, "high_res_sampling_rate": 500.0},
        ),
        "cpsc2018": DatasetMetadata(
            name="China Physiological Signal Challenge 2018 (CPSC 2018)",
            physionet_slug="challenge-2020/1.0.2/training/cpsc_2018",
            description="12-lead ECG recordings from arrhythmia patients with SNOMED-CT clinical codes.",
            sampling_rate_hz=500.0,
            num_leads=12,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            num_records=6877,
            annotation_type="SNOMED-CT diagnostic header annotations",
            source_url="https://physionet.org/content/challenge-2020/1.0.2/",
            is_external=True,
            typical_duration_sec=15.0,
            classes_supported=["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"],
            extra_attributes={"format": "MATLAB v4 .mat with .hea"},
        ),
        "chapman": DatasetMetadata(
            name="Chapman-Shaoxing 12-Lead ECG Database",
            physionet_slug="ecg-arrhythmia/1.0.0",
            description="10,646 12-lead 10-second ECG recordings sampled at 500 Hz.",
            sampling_rate_hz=500.0,
            num_leads=12,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            num_records=10646,
            annotation_type="Rhythm and diagnostic labels",
            source_url="https://physionet.org/content/ecg-arrhythmia/1.0.0/",
            is_external=True,
            typical_duration_sec=10.0,
            classes_supported=["SB", "SR", "AFIB", "ST", "SVT", "AT", "AVNRT", "AVRT", "SA", "AF"],
            extra_attributes={"format": "MATLAB v4 .mat with .hea"},
        ),
        "incartdb": DatasetMetadata(
            name="St Petersburg INCART 12-Lead Arrhythmia Database",
            physionet_slug="incartdb/1.0.0",
            description="75 30-minute 12-lead ambulatory ECG records from patients with suspected coronary artery disease.",
            sampling_rate_hz=257.0,
            num_leads=12,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            num_records=75,
            annotation_type="atr (AAMI EC57 beat annotations)",
            source_url="https://physionet.org/content/incartdb/1.0.0/",
            is_external=True,
            typical_duration_sec=1800.0,
            classes_supported=["N", "S", "V", "F", "Q"],
            extra_attributes={"annotated_leads": 12},
        ),
        "wesad": DatasetMetadata(
            name="WESAD (Wearable Stress and Affect Detection)",
            physionet_slug=None,
            description="Multimodal wearable dataset measuring autonomic stress via RespiBAN chest ECG (700 Hz) and Empatica E4.",
            sampling_rate_hz=700.0,
            num_leads=1,
            lead_names=["ECG_Chest"],
            num_records=15,
            annotation_type="Continuous affective condition (Baseline, Stress, Amusement, Meditation)",
            source_url="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
            is_external=True,
            typical_duration_sec=7200.0,
            classes_supported=["Baseline", "Stress", "Amusement", "Meditation"],
            extra_attributes={"modality": "Continuous autonomic stress / HRV validation"},
        ),
        "workspace_training_corpus": DatasetMetadata(
            name="Workspace SCG Feature Corpus",
            physionet_slug=None,
            description="Local simulated sternal seismocardiography (SCG) mechanical feature corpus with HRV/PRV metrics.",
            sampling_rate_hz=100.0,
            num_leads=1,
            lead_names=["SCG_Z"],
            num_records=60,
            annotation_type="Feature-level tabular CSV with stress_label and arrhythmia_label",
            source_url="local://data/training_features_corpus.csv",
            is_external=False,
            typical_duration_sec=60.0,
            classes_supported=["Normal", "Sinus Tachycardia", "Sinus Bradycardia", "Arrhythmia"],
            extra_attributes={"file_path": "data/training_features_corpus.csv"},
        ),
        "workspace_phyphox_raw": DatasetMetadata(
            name="Workspace Raw Phyphox Test Data",
            physionet_slug=None,
            description="Local 3-axis smartphone accelerometer recording on sternum via Phyphox app.",
            sampling_rate_hz=99.52,
            num_leads=3,
            lead_names=["acc_x", "acc_y", "acc_z"],
            num_records=1,
            annotation_type="Raw inertial time-series with device and epoch metadata",
            source_url="local://data/raw_phyphox_test/",
            is_external=False,
            typical_duration_sec=37.54,
            classes_supported=["Unlabeled mechanical SCG"],
            extra_attributes={"directory_path": "data/raw_phyphox_test/"},
        ),
    }

    @classmethod
    def list_datasets(cls, include_external: bool = True, include_local: bool = True) -> List[str]:
        """List registered dataset identifiers."""
        result = []
        for key, meta in cls._REGISTRY.items():
            if meta.is_external and include_external:
                result.append(key)
            elif not meta.is_external and include_local:
                result.append(key)
        return sorted(result)

    @classmethod
    def get_dataset(cls, name: str) -> DatasetMetadata:
        """Retrieve metadata for a dataset by slug or key."""
        key = name.lower().strip()
        if key in cls._REGISTRY:
            return cls._REGISTRY[key]
        # Match by name or alias
        for k, meta in cls._REGISTRY.items():
            if key == meta.name.lower() or (meta.physionet_slug and key in meta.physionet_slug.lower()):
                return meta
        raise KeyError(f"Dataset '{name}' not found in ECG catalog. Available: {list(cls._REGISTRY.keys())}")

    @classmethod
    def get_de_chazal_split(
        cls, strict_zero_overlap: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Return the de Chazal et al. (2004) inter-patient split for MIT-BIH.

        Args:
            strict_zero_overlap: If True, moves Record 202 to DS1 (because records 201
                and 202 share the exact same human subject, 72yo male).
                If False (default), preserves standard literature partitioning
                (201 in DS1, 202 in DS2) for direct comparison with published papers.

        Returns:
            Tuple of (ds1_records, ds2_records, paced_records)
        """
        ds1 = list(DS1_RECORDS)
        ds2 = list(DS2_RECORDS)
        paced = list(PACED_RECORDS)

        if strict_zero_overlap:
            if "202" in ds2:
                ds2.remove("202")
                ds1.append("202")
                ds1.sort()

        return ds1, ds2, paced

    @classmethod
    def get_patient_mapping(cls) -> Dict[str, str]:
        """
        Return mapping from MIT-BIH record ID to underlying patient/subject ID.
        Identifies subject 201 sharing records 201 and 202.
        """
        mapping = {rec: f"Patient_{rec}" for rec in ALL_MITBIH_RECORDS}
        mapping["202"] = "Patient_201"  # Record 202 is from Patient 201
        return mapping

    @classmethod
    def scan_workspace_assets(cls, workspace_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Scan workspace directory and check existence and health of local data assets.
        """
        root = Path(workspace_root) if workspace_root else Path.cwd()
        results: Dict[str, Any] = {
            "workspace_root": str(root),
            "assets_found": {},
            "external_cache": {},
        }

        # Check local corpus
        corpus_path = root / "data" / "training_features_corpus.csv"
        if corpus_path.exists():
            results["assets_found"]["training_features_corpus"] = {
                "exists": True,
                "path": str(corpus_path),
                "size_bytes": corpus_path.stat().st_size,
            }
        else:
            results["assets_found"]["training_features_corpus"] = {"exists": False}

        # Check phyphox
        phyphox_dir = root / "data" / "raw_phyphox_test"
        if phyphox_dir.exists() and phyphox_dir.is_dir():
            files = [p.name for p in phyphox_dir.iterdir() if not p.name.startswith(".")]
            results["assets_found"]["raw_phyphox_test"] = {
                "exists": True,
                "path": str(phyphox_dir),
                "files": files,
            }
        else:
            results["assets_found"]["raw_phyphox_test"] = {"exists": False}

        # Check benchmark cache directory
        cache_dir = root / "data" / "benchmarks"
        if cache_dir.exists():
            subdirs = [p.name for p in cache_dir.iterdir() if p.is_dir()]
            results["external_cache"] = {
                "exists": True,
                "path": str(cache_dir),
                "cached_databases": subdirs,
            }
        else:
            results["external_cache"] = {"exists": False, "path": str(cache_dir)}

        return results
