"""Evaluation metrics under a deployment prior (positives rare: share = prior).

Validation and test sets keep their natural ~9:1 positive:negative ratio. Every metric below re-weights
the classes so that positives make up `prior` of the total, which is what the deployment looks like.
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def class_weights(y, prior):
    y = np.asarray(y)
    npos, nneg = max(1, (y == 1).sum()), max(1, (y == 0).sum())
    return np.where(y == 1, prior / npos, (1 - prior) / nneg)


def calibrated_prob(z, prior, a=1.0, b=None):
    """Balanced-training logit -> probability. Default is the pure prior shift z + logit(prior)."""
    b = logit(prior) if b is None else b
    return sigmoid(a * np.asarray(z, dtype=np.float64) + b)


def weighted_logloss(p, y, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def weighted_ece(p, y, w, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            wm = w[m].sum()
            conf, acc = float((w[m] * p[m]).sum() / wm), float((w[m] * y[m]).sum() / wm)
            ece += wm / w.sum() * abs(conf - acc)
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": conf, "observed": acc})
    return float(ece), rows


def prior_pr_auc(z, y, prior):
    """Area under the precision-recall curve with precision computed at the deployment prior."""
    fpr, tpr, _ = roc_curve(y, z)
    prec = prior * tpr / np.maximum(prior * tpr + (1 - prior) * fpr, 1e-12)
    prec[0] = prec[1] if len(prec) > 1 else 1.0
    return float(np.sum(np.diff(tpr) * prec[1:]))


def weighted_confusion(p, y, w, thr):
    pred = p >= thr
    tp, fp = w[pred & (y == 1)].sum(), w[pred & (y == 0)].sum()
    fn, tn = w[~pred & (y == 1)].sum(), w[~pred & (y == 0)].sum()
    return tp, fp, fn, tn


def weighted_mcc(p, y, w, thr):
    tp, fp, fn, tn = weighted_confusion(p, y, w, thr)
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-30))
    return float((tp * tn - fp * fn) / den)


def best_threshold(p, y, w):
    cands = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    scores = [weighted_mcc(p, y, w, t) for t in cands]
    i = int(np.argmax(scores))
    return float(cands[i]), float(scores[i])


def evaluate(z, y, prior, a=1.0, b=None, threshold=0.5):
    """z: raw logits from balanced training; y: 0/1 labels. Returns a flat dict of metrics."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.int64)
    out = {"n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        return out
    w = class_weights(y, prior)
    p = calibrated_prob(z, prior, a, b)
    tp, fp, fn, tn = weighted_confusion(p, y, w, threshold)
    out.update({
        "roc_auc": float(roc_auc_score(y, z)),
        "pr_auc_prior": prior_pr_auc(z, y, prior),
        "logloss_prior": weighted_logloss(p, y, w),
        "brier_prior": float((w * (p - y) ** 2).sum() / w.sum()),
        "ece_prior": weighted_ece(p, y, w)[0],
        "threshold": float(threshold),
        "recall": float(tp / max(tp + fn, 1e-30)),
        "specificity": float(tn / max(tn + fp, 1e-30)),
        "precision_prior": float(tp / max(tp + fp, 1e-30)),
        "mcc_prior": weighted_mcc(p, y, w, threshold),
        "mean_p_pos": float(p[y == 1].mean()), "mean_p_neg": float(p[y == 0].mean()),
    })
    return out
