"""Train (or resume) a run: one model per cross-validation fold, then calibration and test evaluation.

    python -m abset_ppi.train --run main                 # start, or resume if runs/main exists
    python -m abset_ppi.train --run main --set train.max_epochs=80
    python -m abset_ppi.train --run smoke --smoke        # 2-minute end-to-end check on a tiny subset

Progress is saved after every epoch (runs/<run>/fold<k>/last.pt). Stop with Ctrl+C or
`python -m abset_ppi.stop --run <run>`; run the same train command again to continue.
"""
import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import finalize
from .data import GraphStore, Normalizer, collate, load_norm_stats, read_records
from .losses import compute_losses
from .metrics import evaluate
from .model import PoseScorer, count_parameters
from .sampler import BalancedResampler
from .utils import (StopFlag, Timer, apply_overrides, atomic_json, atomic_torch_save, fmt_duration, get_logger, load_config,
                    pick_device, read_json, resolve, rng_state, run_dir, save_config, seed_everything, set_rng_state)

METRIC_FIELDS = ["epoch", "seconds", "lr", "train_loss", "train_main", "train_dockq", "train_residue", "train_rank",
                 "val_roc_auc", "val_pr_auc_prior", "val_logloss_prior", "val_brier_prior", "val_ece_prior",
                 "val_mcc_prior", "val_recall", "val_mean_p_pos", "val_mean_p_neg", "best_so_far"]


# ----------------------------------------------------------------------------- helpers shared with finalize

def build_model(cfg, device):
    use_plm = bool(cfg["embeddings"]["use"])
    return PoseScorer(cfg, use_plm, cfg["embeddings"]["dim"]).to(device)


@torch.no_grad()
def predict_logits(model, store, ids, normalizer, cfg, device, batch_size):
    model.eval()
    zs = []
    for i in range(0, len(ids), batch_size):
        b = collate([store[r] for r in ids[i:i + batch_size]], normalizer, cfg["model"]["cross_cutoff"],
                    model.use_plm).to(device)
        zs.append(model(b)["logit"].float().cpu())
    return torch.cat(zs).numpy() if zs else np.zeros(0)


def lr_lambda(total_steps, warmup_frac):
    warm = max(1, int(total_steps * warmup_frac))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.5 * (1 + math.cos(math.pi * prog))
    return f


def metric_is_better(name, new, best):
    if best is None:
        return True
    return new > best if name == "val_roc_auc" else new < best


# ----------------------------------------------------------------------------- one fold

def train_fold(k, cfg, rd, records, store, device, stop, log, status):
    t = cfg["train"]
    fd = rd / f"fold{k}"
    fd.mkdir(parents=True, exist_ok=True)
    split = {r["record_id"]: r["split"] for r in records}
    tr = [r for r in records if r["split"] not in ("test", str(k))]
    va = [r for r in records if r["split"] == str(k)]
    pos = [r["record_id"] for r in tr if r["label"] == 1]
    neg = [r["record_id"] for r in tr if r["label"] == 0]
    pos_by_cx = {r["complex_id"]: r["record_id"] for r in tr if r["label"] == 1}
    matched = {r["record_id"]: pos_by_cx[r["complex_id"]] for r in tr if r["label"] == 0 and r["complex_id"] in pos_by_cx}
    va_ids = [r["record_id"] for r in va]
    va_y = np.array([r["label"] for r in va])

    stats = load_norm_stats(resolve(cfg["data"]["processed_dir"]), f"fold{k}")
    normalizer = Normalizer(stats)
    sampler = BalancedResampler(pos, neg, matched, t["batch_size"], t["matched_pair_prob"], t["seed"] + 1000 * k)
    seed_everything(t["seed"] + k)
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = math.ceil((len(neg) + min(len(neg), len(pos))) / t["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps_per_epoch * t["max_epochs"], t["warmup_frac"]))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    st = {"epoch": 0, "best": None, "best_epoch": None, "bad_epochs": 0, "epoch_times": []}
    last = fd / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        sampler.load_state_dict(ck["sampler"]); set_rng_state(ck["rng"]); st = ck["state"]
        log.info(f"Fold {k}: resuming from saved epoch {st['epoch']} (best {t['early_stop_metric']} = "
                 f"{st['best']:.4f} at epoch {st['best_epoch']})" if st["best"] is not None else
                 f"Fold {k}: resuming from saved epoch {st['epoch']}")
    else:
        log.info(f"Fold {k}: starting. Train {len(pos)} positives / {len(neg)} negatives "
                 f"({len(matched)} negatives with a matching crystal); validation {int(va_y.sum())} / "
                 f"{int((va_y == 0).sum())}. Model has {count_parameters(model):,} parameters; "
                 f"{steps_per_epoch} steps per epoch.")

    metrics_csv = fd / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    prior, metric_name = cfg["calibration"]["prior"], t["early_stop_metric"]
    metric_key = {"val_logloss_prior": "logloss_prior", "val_roc_auc": "roc_auc"}[metric_name]

    def update_status(state):
        status["folds"][str(k)] = {"state": state, "epochs_done": st["epoch"], "max_epochs": t["max_epochs"],
                                   "best_epoch": st["best_epoch"], "best_metric": st["best"],
                                   "patience_left": t["patience"] - st["bad_epochs"]}
        status["current_fold"] = k
        status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_json(status, rd / "status.json")

    update_status("running")
    while st["epoch"] < t["max_epochs"]:
        ep = st["epoch"] + 1
        timer = Timer()
        model.train()
        batches = sampler.epoch_batches()
        sums, n_steps = {}, 0
        for step, ids in enumerate(batches, 1):
            if stop.check():
                log.info(f"Fold {k}: stopped during epoch {ep}. Progress up to epoch {st['epoch']} is saved; "
                         f"epoch {ep} will be redone on resume.")
                update_status("stopped")
                return "stopped"
            b = collate([store[r] for r in ids], normalizer, cfg["model"]["cross_cutoff"], model.use_plm).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(b)
                loss, parts = compute_losses(out, b, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            for key in ("loss", "main", "dockq", "residue", "rank"):
                sums[key] = sums.get(key, 0.0) + parts[key]
            n_steps += 1
            if t["log_every"] and step % t["log_every"] == 0:
                log.info(f"  fold {k} | epoch {ep}/{t['max_epochs']} | step {step}/{len(batches)} | "
                         f"loss {sums['loss'] / n_steps:.4f}")
        train_avg = {key: v / max(1, n_steps) for key, v in sums.items()}

        z = predict_logits(model, store, va_ids, normalizer, cfg, device, cfg["eval"]["batch_size"])
        m = evaluate(z, va_y, prior)
        value = m[metric_key]
        improved = metric_is_better(metric_name, value, st["best"])
        if improved:
            st.update(best=float(value), best_epoch=ep, bad_epochs=0)
            atomic_torch_save({"model": model.state_dict(), "config": cfg, "norm_stats": stats, "fold": k,
                               "epoch": ep, "val_metrics": m}, fd / "best.pt")
        else:
            st["bad_epochs"] += 1
        st["epoch"] = ep
        st["epoch_times"] = (st["epoch_times"] + [timer.elapsed()])[-20:]
        atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "scaler": scaler.state_dict(), "sampler": sampler.state_dict(), "rng": rng_state(),
                           "state": st, "config": cfg}, last)
        row = {"epoch": ep, "seconds": round(timer.elapsed(), 1), "lr": opt.param_groups[0]["lr"],
               **{f"train_{key}": round(v, 5) for key, v in train_avg.items()},
               **{f"val_{key}": round(m.get(key, float("nan")), 5) for key in
                  ("roc_auc", "pr_auc_prior", "logloss_prior", "brier_prior", "ece_prior", "mcc_prior", "recall",
                   "mean_p_pos", "mean_p_neg")},
               "best_so_far": int(improved)}
        with open(metrics_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow({kk: row.get(kk) for kk in METRIC_FIELDS})

        remaining = t["max_epochs"] - ep
        eta = np.mean(st["epoch_times"]) * min(remaining, t["patience"] - st["bad_epochs"])
        verdict = "NEW BEST" if improved else f"no improvement {st['bad_epochs']}/{t['patience']}"
        log.info(f"Fold {k} | epoch {ep}/{t['max_epochs']} | {fmt_duration(timer.elapsed())} | "
                 f"train loss {train_avg.get('loss', float('nan')):.4f} | val AUC {m.get('roc_auc', float('nan')):.4f} "
                 f"| val logloss@prior {m.get('logloss_prior', float('nan')):.4f} | {verdict} | "
                 f"fold time left <= {fmt_duration(eta)}")
        update_status("running")
        if st["bad_epochs"] >= t["patience"]:
            log.info(f"Fold {k}: early stopping, no improvement for {t['patience']} epochs.")
            break
        if stop.check():
            log.info(f"Fold {k}: stop requested; epoch {ep} saved. Run the same train command to resume.")
            update_status("stopped")
            return "stopped"

    best = torch.load(fd / "best.pt", map_location="cpu", weights_only=False)
    atomic_json({"fold": k, "epochs_run": st["epoch"], "best_epoch": st["best_epoch"], "best_metric_name": metric_name,
                 "best_metric": st["best"], "val_metrics_at_best": best["val_metrics"]}, fd / "done.json")
    update_status("done")
    log.info(f"Fold {k}: finished. Best epoch {st['best_epoch']}: val AUC {best['val_metrics'].get('roc_auc', float('nan')):.4f}, "
             f"val logloss@prior {best['val_metrics'].get('logloss_prior', float('nan')):.4f}")
    return "done"


# ----------------------------------------------------------------------------- main

def smoke_subset(records):
    keep, count = [], {}
    for r in records:
        key = (r["split"], r["label"])
        lim = 24 if r["label"] == 1 else 8
        if count.get(key, 0) < lim:
            keep.append(r); count[key] = count.get(key, 0) + 1
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/<run>/")
    ap.add_argument("--config", default=None, help="YAML config for a NEW run (default config/default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values for a NEW run")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny quick run to verify the setup (use a separate run name)")
    ap.add_argument("--no-finalize", action="store_true", help="do not calibrate / evaluate after the folds finish")
    a = ap.parse_args(argv)

    rd = run_dir(a.run)
    frozen = rd / "config.yaml"
    rd.mkdir(parents=True, exist_ok=True)
    log = get_logger(rd / "train.log")
    if frozen.exists():
        cfg = load_config(frozen)
        if a.config or a.set or a.smoke:
            log.info("NOTE: this run already exists, so its saved config is used and --config/--set/--smoke are ignored. "
                     "Use a new --run name to train with different settings.")
        smoke = bool(cfg.get("_smoke"))
    else:
        cfg = load_config(a.config)
        if a.smoke:
            cfg["_smoke"] = True
            cfg["train"].update(max_epochs=2, patience=2, folds=[0], log_every=2)
        cfg = apply_overrides(cfg, a.set)                 # explicit --set values win over smoke defaults
        save_config(cfg, frozen)
        smoke = a.smoke

    stop = StopFlag(rd / "STOP")
    if stop.stop_file.exists():
        log.info("Removing an old STOP file from a previous stop request.")
        stop.clear()
    device = pick_device(a.device)
    log.info("=" * 100)
    log.info(f"Run '{a.run}' on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
             + (" [SMOKE TEST]" if smoke else ""))
    status = read_json(rd / "status.json", {"run": a.run, "folds": {}})
    status.update(state="training")
    status.pop("error", None)
    atomic_json(status, rd / "status.json")

    try:
        processed = resolve(cfg["data"]["processed_dir"])
        records = read_records(processed)
        if smoke:
            records = smoke_subset(records)
        emb = cfg["embeddings"]
        log.info(f"Loading {len(records)} graphs" + (f" and {emb['model']} embeddings" if emb["use"] else " (no embeddings)") + " ...")
        store = GraphStore(processed, records, resolve(cfg["data"]["embeddings_dir"]),
                           emb["model"] if emb["use"] else None, log)
        folds = [int(f) for f in cfg["train"]["folds"]]
        for k in folds:
            if (rd / f"fold{k}" / "done.json").exists():
                log.info(f"Fold {k}: already finished, skipping.")
                continue
            if train_fold(k, cfg, rd, records, store, device, stop, log, status) == "stopped":
                status["state"] = "stopped"
                atomic_json(status, rd / "status.json")
                stop.clear()
                log.info(f"Training stopped cleanly. Resume with:  python -m abset_ppi.train --run {a.run}")
                return 0
        log.info(f"All folds {folds} finished.")
        if a.no_finalize:
            status["state"] = "folds_done"
            atomic_json(status, rd / "status.json")
            log.info(f"Skipping finalize. Run it with:  python -m abset_ppi.finalize --run {a.run}")
            return 0
        finalize.run(rd, cfg, store, records, device, log)
        status["state"] = "finished"
        atomic_json(status, rd / "status.json")
        log.info(f"Run '{a.run}' complete. See runs/{a.run}/test_metrics.json; score complexes with "
                 f"python -m abset_ppi.predict --run {a.run} ...")
        return 0
    except Exception as e:
        status["state"] = "failed"
        status["error"] = f"{type(e).__name__}: {e}"
        atomic_json(status, rd / "status.json")
        log.error("Training failed:\n" + traceback.format_exc())
        log.error(f"Progress up to the last completed epoch is saved. After fixing the problem, resume with: "
                  f"python -m abset_ppi.train --run {a.run}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
