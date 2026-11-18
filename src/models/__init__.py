"""
ECG Rhythm Classification Models Package.
Provides unified model interfaces and candidate architectures (Methods A, B, C).
"""

from src.models.base_classifier import BaseECGClassifier
from src.models.method_a_gbdt import MethodAGBDTClassifier, extract_features
from src.models.method_b_resnet import (
    ECGResNet1D,
    MethodBResNetClassifier,
    MultiClassFocalLoss,
)
from src.models.method_c_hybrid_attn import (
    BahdanauTemporalAttention,
    CNNBiLSTMAttention1D,
    MethodCHybridAttentionClassifier,
)

__all__ = [
    "BaseECGClassifier",
    "MethodAGBDTClassifier",
    "extract_features",
    "MethodBResNetClassifier",
    "ECGResNet1D",
    "MultiClassFocalLoss",
    "MethodCHybridAttentionClassifier",
    "CNNBiLSTMAttention1D",
    "BahdanauTemporalAttention",
]
