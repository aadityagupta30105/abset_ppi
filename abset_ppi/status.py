"""Show progress and metrics of a run.

    python -m abset_ppi.status --run main             # summary table
    python -m abset_ppi.status --run main --watch 60  # refresh every 60 s (Ctrl+C to leave; training continues)
    python -m abset_ppi.status --run main --plot      # save learning curves to runs/main/learning_curves.png
    python -m abset_ppi.status                        # list all runs
"""
import argparse
import csv
import sys
import time

from .utils import RUNS_DIR, load_config, read_json, run_dir


def read_metrics(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def fnum(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def show(name):
    rd = run_dir(name)
    if not (rd / "config.yaml").exists():
        raise SystemExit(f"No run named '{name}' in {RUNS_DIR}")
    cfg = load_config(rd / "config.yaml")
    st = read_json(rd / "status.json", {})
    folds = [int(f) for f in cfg["train"]["folds"]]
    print(f"\nRun '{name}'   state: {st.get('state', 'not started')}   last update: {st.get('updated', '-')}")
    if st.get("error"):
        print(f"  last error: {st['error']}  (details in runs/{name}/train.log)")
    print(f"  prior for calibration: {cfg['calibration']['prior']}   early-stop metric: {cfg['train']['early_stop_metric']}"
          f"   embeddings: {cfg['embeddings']['model'] if cfg['embeddings']['use'] else 'off'}")
    hdr = f"  {'fold':<5}{'state':<10}{'epochs':>10}{'best ep':>9}{'best val':>10}{'val AUC':>9}{'val PR@p':>10}{'last ep':>9}{'last AUC':>10}{'s/epoch':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in folds:
        rows = read_metrics(rd / f"fold{k}" / "metrics.csv")
        done = read_json(rd / f"fold{k}" / "done.json")
        fs = st.get("folds", {}).get(str(k), {})
        state = "done" if done else fs.get("state", "pending")
        best = [r for r in rows if r["epoch"] == str(fs.get("best_epoch"))] or ([max(rows, key=lambda r: -float(r["val_logloss_prior"]))] if rows else [])
        b = best[0] if best else {}
        last = rows[-1] if rows else {}
        secs = sum(float(r["seconds"]) for r in rows[-10:]) / max(1, len(rows[-10:])) if rows else None
        print(f"  {k:<5}{state:<10}{(str(len(rows)) + '/' + str(cfg['train']['max_epochs'])):>10}"
              f"{(b.get('epoch') or '-'):>9}{fnum(fs.get('best_metric')):>10}{fnum(b.get('val_roc_auc')):>9}"
              f"{fnum(b.get('val_pr_auc_prior')):>10}{(last.get('epoch') or '-'):>9}{fnum(last.get('val_roc_auc')):>10}"
              f"{fnum(secs, 1):>9}")
    rep = read_json(rd / "test_metrics.json")
    if rep:
        cv = rep["cross_validation_mean_std"]
        print("\n  Cross-validation (mean +/- sd over folds): " + ", ".join(
            f"{k} {v['mean']:.4f}+/-{v['std']:.4f}" for k, v in cv.items()))
        t = rep.get("test", {})
        if t:
            print(f"  TEST  ({t['n_pos']} pos / {t['n_neg']} neg): ROC-AUC {t['roc_auc']:.4f} | PR-AUC@prior {t['pr_auc_prior']:.4f}"
                  f" | logloss@prior {t['logloss_prior']:.4f} | ECE@prior {t['ece_prior']:.4f} | recall {t['recall']:.3f}"
                  f" | precision@prior {t['precision_prior']:.3f} | threshold {t['threshold']:.3f}")
        print(f"  Model file: runs/{name}/final_model.pt")
    elif all((rd / f"fold{k}" / "done.json").exists() for k in folds):
        print(f"\n  All folds done but not finalized. Run:  python -m abset_ppi.finalize --run {name}")
    if (rd / "STOP").exists():
        print("\n  A STOP request is pending (the trainer will exit after its current step).")


def plot(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rd = run_dir(name)
    cfg = load_config(rd / "config.yaml")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k in [int(f) for f in cfg["train"]["folds"]]:
        rows = read_metrics(rd / f"fold{k}" / "metrics.csv")
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        for ax, key in zip(axes, ("train_loss", "val_logloss_prior", "val_roc_auc")):
            ax.plot(ep, [float(r[key]) for r in rows], label=f"fold {k}")
    for ax, title in zip(axes, ("training loss", "validation log-loss at prior (lower is better)", "validation ROC-AUC")):
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    out = rd / "learning_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"Saved {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if not a.run:
        runs = sorted(p.name for p in RUNS_DIR.glob("*") if (p / "config.yaml").exists()) if RUNS_DIR.exists() else []
        print("Runs:", ", ".join(runs) if runs else "none yet")
        for r in runs:
            print(f"  {r}: {read_json(RUNS_DIR / r / 'status.json', {}).get('state', 'not started')}")
        return 0
    if a.plot:
        plot(a.run)
        return 0
    try:
        while True:
            show(a.run)
            if not a.watch:
                break
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
