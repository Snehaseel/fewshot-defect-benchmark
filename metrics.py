"""Evaluation metrics.

Two operating thresholds are reported for triage:
  * oracle:   the 95th percentile of scores on the test set's GOOD images. This uses test
              labels, so it is an upper bound that a real returns centre could not achieve.
  * deployable: the maximum leave-one-out score on the k TRAINING images. It uses no test
              information; we report both the recall and the false-positive rate it actually
              achieves on the test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

TARGET_FPR = 0.05


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def pixel_auroc(maps: np.ndarray | None, masks: np.ndarray) -> float:
    if maps is None:
        return float("nan")
    y = masks.reshape(-1).astype(np.uint8)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, maps.reshape(-1).astype(np.float32)))


def oracle_recall(scores: np.ndarray, labels: np.ndarray, fpr: float = TARGET_FPR) -> float:
    thr = np.quantile(scores[labels == 0], 1.0 - fpr)
    return float(np.mean(scores[labels == 1] > thr))


def recall_fpr_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> tuple[float, float]:
    recall = float(np.mean(scores[labels == 1] > thr))
    fpr = float(np.mean(scores[labels == 0] > thr))
    return recall, fpr


def deployable_threshold(loo_scores: np.ndarray | None) -> float:
    if loo_scores is None or len(loo_scores) == 0:
        return float("nan")
    return float(np.max(loo_scores))


def evaluate(scores, maps, labels, masks, loo_scores=None) -> dict:
    out = {
        "image_auroc": image_auroc(scores, labels),
        "pixel_auroc": pixel_auroc(maps, masks),
        "recall_oracle_5fpr": oracle_recall(scores, labels),
    }
    thr = deployable_threshold(loo_scores)
    if np.isnan(thr):
        out["recall_deploy"], out["fpr_deploy"] = float("nan"), float("nan")
    else:
        out["recall_deploy"], out["fpr_deploy"] = recall_fpr_at(scores, labels, thr)
    return out
