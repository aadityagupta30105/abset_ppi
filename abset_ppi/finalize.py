"""Calibrate the fold ensemble and evaluate it once on the held-out test set.

Runs automatically at the end of `train`. Run it yourself with:
    python -m abset_ppi.finalize --run main

Steps
 1. Each fold's best model scores its own validation fold -> pooled out-of-fold (OOF) logits.
 2. Fit p = sigmoid(a*z + b) on the OOF logits with class weights that re-create the deployment prior.
 3. Pick the decision threshold that maximises prior-weighted MCC on the OOF predictions.
 4. Test set: average the logits of all fold models, apply the calibration, report metrics.
 5. Export runs/<run>/final_model.pt, the single file used by `predict`.
"""
import argparse
import csv
import sys

import numpy as np
import torch
from scipy.optimize import minimize

from .data import GraphStore, Normalizer, read_records
from .metrics import (best_threshold, calibrated_prob, class_weights, evaluate, logit, weighted_ece)
from .utils import atomic_json, atomic_torch_save, get_logger, load_config, pick_device, resolve, run_dir


def fit_platt(z, y, prior):
    w = class_weights(y, prior)
    def nll(params):
        a, b = params
        p = np.clip(1 / (1 + np.exp(-(a * z + b))), 1e-7, 1 - 1e-7)
        # weak pull toward the pure prior shift (a=1): keeps the slope finite when OOF classes separate perfectly
        return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum() + 1e-3 * (a - 1) ** 2
    res = minimize(nll, x0=[1.0, logit(prior)], method="L-BFGS-B", bounds=[(0.05, 10.0), (-30.0, 30.0)])
    return float(res.x[0]), float(res.x[1]), bool(res.success)


def load_fold_model(path, device):
    from .train import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"], device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Normalizer(ck["norm_stats"]), ck


def run(rd, cfg, store, records, device, log):
    from .train import predict_logits
    folds = [int(f) for f in cfg["train"]["folds"]]
    missing = [k for k in folds if not (rd / f"fold{k}" / "done.json").exists()]
    if missing:
        raise SystemExit(f"Folds {missing} are not finished yet; finish training first.")
    prior = cfg["calibration"]["prior"]
    bs = cfg["eval"]["batch_size"]
    test = [r for r in records if r["split"] == "test"]
    test_ids, test_y = [r["record_id"] for r in test], np.array([r["label"] for r in test])

    oof_ids, oof_z, oof_y, oof_fold, test_z, export = [], [], [], [], [], []
    for k in folds:
        model, normalizer, ck = load_fold_model(rd / f"fold{k}" / "best.pt", device)
        va = [r for r in records if r["split"] == str(k)]
        ids = [r["record_id"] for r in va]
        oof_ids += ids; oof_y += [r["label"] for r in va]; oof_fold += [k] * len(ids)
        oof_z.append(predict_logits(model, store, ids, normalizer, cfg, device, bs))
        test_z.append(predict_logits(model, store, test_ids, normalizer, cfg, device, bs))
        export.append({"fold": k, "best_epoch": ck["epoch"], "state_dict": {n: t.cpu() for n, t in ck["model"].items()},
                       "norm_stats": ck["norm_stats"]})
        log.info(f"Finalize: scored fold {k} validation ({len(ids)}) and test ({len(test_ids)}) records")
    oof_z, oof_y = np.concatenate(oof_z), np.array(oof_y)
    a, b, ok = fit_platt(oof_z, oof_y, prior)
    p_oof = calibrated_prob(oof_z, prior, a, b)
    thr, mcc = best_threshold(p_oof, oof_y, class_weights(oof_y, prior))
    calib = {"prior": prior, "a": a, "b": b, "fit_converged": ok, "threshold": thr, "oof_mcc_at_threshold": mcc,
             "formula": "p = 1 / (1 + exp(-(a * mean_fold_logit + b)))"}
    log.info(f"Finalize: calibration a={a:.3f}, b={b:.3f} (prior {prior}); decision threshold {thr:.3f}")

    z_test = np.mean(np.stack(test_z), 0) if test_ids else np.zeros(0)
    p_test = calibrated_prob(z_test, prior, a, b)
    per_fold_cv = {}
    for k in folds:
        m = np.array(oof_fold) == k
        per_fold_cv[str(k)] = evaluate(oof_z[m], oof_y[m], prior)
    cv_keys = ["roc_auc", "pr_auc_prior", "logloss_prior", "brier_prior"]
    cv_summary = {key: {"mean": float(np.nanmean([per_fold_cv[f].get(key, np.nan) for f in per_fold_cv])),
                        "std": float(np.nanstd([per_fold_cv[f].get(key, np.nan) for f in per_fold_cv]))} for key in cv_keys}
    report = {
        "cross_validation_per_fold": per_fold_cv,
        "cross_validation_mean_std": cv_summary,
        "oof_calibrated": evaluate(oof_z, oof_y, prior, a, b, thr),
        "test": evaluate(z_test, test_y, prior, a, b, thr) if len(test_ids) else {},
        "calibration": calib,
        "test_reliability": weighted_ece(p_test, test_y, class_weights(test_y, prior))[1] if len(test_ids) else [],
        "notes": "All *_prior metrics weight classes to the deployment prior. Test has few negatives; treat as indicative.",
    }
    atomic_json(calib, rd / "calibration.json")
    atomic_json(report, rd / "test_metrics.json")
    with open(rd / "oof_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["record_id", "fold", "label", "logit", "probability"])
        w.writerows(zip(oof_ids, oof_fold, oof_y.tolist(), np.round(oof_z, 5), np.round(p_oof, 6)))
    with open(rd / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["record_id", "label", "mean_logit", "probability", "predicted_binding"])
        w.writerows(zip(test_ids, test_y.tolist(), np.round(z_test, 5), np.round(p_test, 6), (p_test >= thr).astype(int)))
    atomic_torch_save({"format": "abset_ppi.final_model.v1", "config": cfg, "folds": export, "calibration": calib},
                      rd / "final_model.pt")
    t = report["test"]
    if t:
        log.info(f"TEST ({t['n_pos']} positives / {t['n_neg']} negatives): ROC-AUC {t['roc_auc']:.4f} | "
                 f"PR-AUC@prior {t['pr_auc_prior']:.4f} | logloss@prior {t['logloss_prior']:.4f} | "
                 f"recall {t['recall']:.3f} | precision@prior {t['precision_prior']:.3f}")
    log.info(f"Finalize: wrote calibration.json, test_metrics.json, oof/test predictions and final_model.pt in {rd}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    rd = run_dir(a.run)
    if not (rd / "config.yaml").exists():
        raise SystemExit(f"No run named '{a.run}' in {rd.parent}")
    cfg = load_config(rd / "config.yaml")
    log = get_logger(rd / "train.log")
    records = read_records(resolve(cfg["data"]["processed_dir"]))
    if cfg.get("_smoke"):
        from .train import smoke_subset
        records = smoke_subset(records)
    emb = cfg["embeddings"]
    store = GraphStore(resolve(cfg["data"]["processed_dir"]), records, resolve(cfg["data"]["embeddings_dir"]),
                       emb["model"] if emb["use"] else None, log)
    run(rd, cfg, store, records, pick_device(a.device), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
