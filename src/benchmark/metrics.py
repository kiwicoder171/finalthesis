"""
ANSI/AAMI EC57 Clinical Evaluation Metrics Engine (Contract 3).
Computes 5x5 confusion matrices, class-specific Sensitivity (Se), Positive Predictivity (+P),
Specificity (Sp), F1-score, Macro F1, and Gross vs. Record-Average aggregations.
"""

from typing import Any, Dict, List, Optional
import numpy as np

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    Compute C x C confusion matrix where M[i, j] is count of true class i predicted as class j.
    """
    y_t_raw = np.asarray(y_true).ravel()
    y_p_raw = np.asarray(y_pred).ravel()
    if len(y_t_raw) != len(y_p_raw):
        raise ValueError(f"Length mismatch: y_true={len(y_t_raw)}, y_pred={len(y_p_raw)}")
    if len(y_t_raw) > 0:
        if not (np.issubdtype(y_t_raw.dtype, np.integer) or np.all(np.equal(np.mod(y_t_raw, 1), 0))):
            raise ValueError("y_true must contain integer class labels")
        if not (np.issubdtype(y_p_raw.dtype, np.integer) or np.all(np.equal(np.mod(y_p_raw, 1), 0))):
            raise ValueError("y_pred must contain integer class labels")
    y_t = y_t_raw.astype(int)
    y_p = y_p_raw.astype(int)
    if len(y_t) > 0:
        if np.any((y_t < 0) | (y_t >= num_classes)):
            raise ValueError(f"y_true labels out of range [0, {num_classes - 1}]")
        if np.any((y_p < 0) | (y_p >= num_classes)):
            raise ValueError(f"y_pred labels out of range [0, {num_classes - 1}]")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_t, y_p), 1)
    return cm


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """
    Extract per-class and aggregate metrics from a 5x5 confusion matrix.
    Handles zero-division edge cases gracefully per ANSI/AAMI EC57.
    """
    total = int(cm.sum())
    class_metrics: Dict[str, Dict[str, float]] = {}
    f1_all: List[float] = []
    f1_present: List[float] = []
    f1_4class: List[float] = []

    for idx, cname in enumerate(AAMI_CLASSES):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)

        support = tp + fn
        pred_pos = tp + fp

        # Sensitivity (Recall)
        if support > 0:
            se = float(tp / support)
        else:
            se = 0.0

        # Positive Predictivity (+P / Precision)
        if pred_pos > 0:
            ppv = float(tp / pred_pos)
        else:
            ppv = 0.0

        # Specificity (True Negative Rate)
        if (tn + fp) > 0:
            sp = float(tn / (tn + fp))
        else:
            sp = 1.0

        # F1-Score
        if (2 * tp + fp + fn) > 0:
            f1 = float((2.0 * tp) / (2.0 * tp + fp + fn))
        else:
            f1 = 0.0

        class_metrics[cname] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "support": support,
            "Se": round(se, 6),
            "+P": round(ppv, 6),
            "Sp": round(sp, 6),
            "F1": round(f1, 6),
        }

        f1_all.append(f1)
        if support > 0:
            f1_present.append(f1)
        if idx < 4:  # N, S, V, F
            f1_4class.append(f1)

    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    # Use active classes with support > 0 for macro_f1 so datasets with 0 'Q' beats aren't penalized by 0/0
    macro_f1 = float(np.mean(f1_present)) if f1_present else 0.0
    macro_f1_5class = float(np.mean(f1_all))
    macro_f1_4class = float(np.mean(f1_4class))

    return {
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics,
        "macro_f1": round(macro_f1, 6),
        "macro_f1_5class": round(macro_f1_5class, 6),
        "macro_f1_4class": round(macro_f1_4class, 6),
        "accuracy": round(accuracy, 6),
        "total_beats": total,
    }


def evaluate_ec57_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    record_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate predictions under ANSI/AAMI EC57:1998/(R)2008 standard (Contract 3).

    Computes both Gross Statistics (pooled across all beats) and Average Statistics
    (per-record means, excluding records with 0 ground-truth support for class c per EC57 5.2.2).
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()
    cm_gross = compute_confusion_matrix(y_t, y_p, num_classes=5)
    gross = _metrics_from_cm(cm_gross)

    result: Dict[str, Any] = {
        "confusion_matrix": cm_gross,
        "class_metrics": gross["class_metrics"],
        "macro_f1": gross["macro_f1"],
        "macro_f1_5class": gross["macro_f1_5class"],
        "macro_f1_4class": gross["macro_f1_4class"],
        "accuracy": gross["accuracy"],
        "total_beats": gross["total_beats"],
        "gross_metrics": gross,
        "average_metrics": None,
    }

    if record_ids is not None:
        rec_arr = np.asarray(record_ids).ravel()
        if len(rec_arr) != len(y_t):
            raise ValueError(f"record_ids length {len(rec_arr)} != y_true length {len(y_t)}")
        unique_recs = np.unique(rec_arr)
        per_class_se: Dict[str, List[float]] = {c: [] for c in AAMI_CLASSES}
        per_class_ppv: Dict[str, List[float]] = {c: [] for c in AAMI_CLASSES}
        per_class_sp: Dict[str, List[float]] = {c: [] for c in AAMI_CLASSES}
        per_class_f1: Dict[str, List[float]] = {c: [] for c in AAMI_CLASSES}

        for rec in unique_recs:
            mask = rec_arr == rec
            cm_rec = compute_confusion_matrix(y_t[mask], y_p[mask], num_classes=5)
            rec_stats = _metrics_from_cm(cm_rec)
            for cname in AAMI_CLASSES:
                c_stat = rec_stats["class_metrics"][cname]
                # Per ANSI/AAMI EC57 clause 5.2.2: include record in Se/F1 average if support > 0
                if c_stat["support"] > 0:
                    per_class_se[cname].append(c_stat["Se"])
                    per_class_f1[cname].append(c_stat["F1"])
                if (c_stat["TP"] + c_stat["FP"]) > 0:
                    per_class_ppv[cname].append(c_stat["+P"])
                if (c_stat["TN"] + c_stat["FP"]) > 0:
                    per_class_sp[cname].append(c_stat["Sp"])

        avg_class_metrics: Dict[str, Dict[str, float]] = {}
        avg_f1s: List[float] = []
        for cname in AAMI_CLASSES:
            se_avg = float(np.mean(per_class_se[cname])) if per_class_se[cname] else 0.0
            ppv_avg = float(np.mean(per_class_ppv[cname])) if per_class_ppv[cname] else 0.0
            sp_avg = float(np.mean(per_class_sp[cname])) if per_class_sp[cname] else 1.0
            f1_avg = float(np.mean(per_class_f1[cname])) if per_class_f1[cname] else 0.0
            avg_class_metrics[cname] = {
                "Se_avg": round(se_avg, 6),
                "+P_avg": round(ppv_avg, 6),
                "Sp_avg": round(sp_avg, 6),
                "F1_avg": round(f1_avg, 6),
                "records_with_support": len(per_class_se[cname]),
            }
            if per_class_f1[cname]:
                avg_f1s.append(f1_avg)

        result["average_metrics"] = {
            "num_records": len(unique_recs),
            "class_metrics": avg_class_metrics,
            "macro_f1_avg": round(float(np.mean(avg_f1s)) if avg_f1s else 0.0, 6),
        }

    return result
