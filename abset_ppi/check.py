"""Check that the environment and the data are ready for training.

    python -m abset_ppi.check

Prints versions, GPU, data counts and embedding coverage, then runs one forward/backward pass on the
selected device. Ends with READY or a list of problems.
"""
import argparse
import shutil
import sys
import time

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    problems, notes = [], []
    print(f"Python  {sys.version.split()[0]}")
    print(f"PyTorch {torch.__version__}  (CUDA build: {torch.version.cuda})")
    for mod in ("numpy", "scipy", "sklearn", "Bio", "yaml", "transformers", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod:<13}{getattr(m, '__version__', 'ok')}")
        except ImportError:
            (notes if mod == "matplotlib" else problems).append(f"package '{mod}' is not installed")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GB")
    else:
        notes.append("no CUDA GPU visible to PyTorch: training would run on CPU (slow). "
                     "If the machine has an NVIDIA GPU, install the CUDA build of PyTorch (see README).")

    from .data import GraphStore, Normalizer, collate, embedding_path, load_norm_stats, read_records
    from .utils import ROOT, load_config, pick_device, resolve
    cfg = load_config(a.config)
    processed = resolve(cfg["data"]["processed_dir"])
    free = shutil.disk_usage(ROOT).free / 2**30
    print(f"Package folder: {ROOT}  (free disk {free:.0f} GB)")
    if free < 5:
        problems.append(f"only {free:.1f} GB free disk; embeddings need ~2.5 GB and checkpoints ~0.2 GB")
    try:
        records = read_records(processed)
    except FileNotFoundError:
        problems.append(f"{processed / 'records.csv'} not found")
        records = []
    if records:
        from collections import Counter
        missing = [r for r in records if not (processed / r["graph"]).exists()]
        c = Counter((r["split"], r["label"]) for r in records)
        print(f"Data: {len(records)} records, {len(records) - len(missing)} graph files present")
        print("  split   positives  negatives")
        for s in ["test", "0", "1", "2", "3", "4"]:
            print(f"  {s:<8}{c[(s, 1)]:>9}{c[(s, 0)]:>11}")
        if missing:
            problems.append(f"{len(missing)} graph files are missing (first {missing[0]['graph']})")
        if cfg["embeddings"]["use"]:
            emb_root = resolve(cfg["data"]["embeddings_dir"])
            have = sum(embedding_path(emb_root, cfg["embeddings"]["model"], r["record_id"]).exists() for r in records)
            print(f"Embeddings ({cfg['embeddings']['model']}): {have}/{len(records)} records")
            if have < len(records):
                notes.append(f"embeddings missing for {len(records) - have} records: run  python -m abset_ppi.embed  "
                             "before training (or train with --set embeddings.use=false)")
    if records and not problems:
        device = pick_device(a.device)
        use_plm = cfg["embeddings"]["use"] and not any("embeddings missing" in n for n in notes)
        cfg["embeddings"]["use"] = use_plm
        sample = [r for r in records if r["split"] == "0"][:4] + [r for r in records if r["label"] == 0][:4]
        store = GraphStore(processed, sample, resolve(cfg["data"]["embeddings_dir"]),
                           cfg["embeddings"]["model"] if use_plm else None)
        from .losses import compute_losses
        from .train import build_model
        model = build_model(cfg, device)
        b = collate([store[r["record_id"]] for r in sample], Normalizer(load_norm_stats(processed, "fold0")),
                    cfg["model"]["cross_cutoff"], use_plm).to(device)
        t0 = time.time()
        out = model(b)
        loss, _ = compute_losses(out, b, cfg)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"Test forward/backward on {device} with {len(sample)} graphs"
              f"{' (with embeddings)' if use_plm else ' (without embeddings)'}: OK, loss {float(loss):.3f}, "
              f"{time.time() - t0:.2f} s")
    print()
    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("READY" if not problems else "NOT READY: fix the problems above")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
