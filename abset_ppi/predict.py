"""Score antibody–antigen complexes with a trained run.

Output: probability (0–1) that the complex is in a native, interacting orientation, calibrated for a
deployment where such orientations are rare (the run's calibration.prior), plus a yes/no call at the
run's decision threshold.

Dataset records (already featurised):
    python -m abset_ppi.predict --run main --split test
    python -m abset_ppi.predict --run main --records exp:1a2y_1,dbl:1n8z_1

New complexes (PDB + AbSet-format descriptor CSV with the same base name):
    python -m abset_ppi.predict --run main --pdb my.pdb --csv my.csv
    python -m abset_ppi.predict --run main --input-dir folder_with_pdb_and_csv_pairs
    # chains must be H (heavy), L (light) and one antigen chain; otherwise map them:
    python -m abset_ppi.predict --run main --pdb my.pdb --csv my.csv --heavy B --light C --antigen A

Results are printed and written to --out (default runs/<run>/predictions_<timestamp>.csv).
"""
import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data import Normalizer, collate, embedding_path, read_records, slim_graph
from .metrics import calibrated_prob
from .utils import pick_device, resolve, run_dir


def load_final(path, device):
    from .train import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != "abset_ppi.final_model.v1":
        raise SystemExit(f"{path} is not a final_model.pt from abset_ppi.finalize")
    models = []
    for f in ck["folds"]:
        m = build_model(ck["config"], device)
        m.load_state_dict(f["state_dict"]); m.eval()
        models.append((f["fold"], m, Normalizer(f["norm_stats"])))
    return ck["config"], ck["calibration"], models


@torch.no_grad()
def score(graphs, cfg, calib, models, device, batch_size=16):
    per_fold = []
    for _, model, norm in models:
        zs = []
        for i in range(0, len(graphs), batch_size):
            b = collate(graphs[i:i + batch_size], norm, cfg["model"]["cross_cutoff"], model.use_plm).to(device)
            zs.append(model(b)["logit"].float().cpu())
        per_fold.append(torch.cat(zs).numpy())
    per_fold = np.stack(per_fold)
    z = per_fold.mean(0)
    p = calibrated_prob(z, calib["prior"], calib["a"], calib["b"])
    return z, p, per_fold


def remap_chains(pdb, csv_path, mapping, tmpdir):
    """Rewrite chain letters so the antibody chains are H and L and the antigen is A."""
    out_pdb, out_csv = Path(tmpdir) / Path(pdb).name, Path(tmpdir) / Path(csv_path).name
    with open(pdb) as f, open(out_pdb, "w") as g:
        for line in f:
            if line.startswith(("ATOM", "HETATM", "TER", "ANISOU")) and len(line) > 21:
                c = line[21]
                if c in mapping:
                    line = line[:21] + mapping[c] + line[22:]
                elif line.startswith(("ATOM", "HETATM")):
                    continue                              # drop chains that are not part of the pair
            g.write(line)
    with open(csv_path, newline="") as f, open(out_csv, "w", newline="") as g:
        rd = csv.DictReader(f); wr = csv.DictWriter(g, fieldnames=rd.fieldnames); wr.writeheader()
        for r in rd:
            if r["chain"] in mapping:
                r["chain"] = mapping[r["chain"]]; wr.writerow(r)
    return out_pdb, out_csv


def featurise_new(pairs, mapping, cfg, device):
    from .featurize import build_graph
    graphs, ids, failed = [], [], {}
    embedder = None
    if cfg["embeddings"]["use"]:
        from .embed import Embedder
        embedder = Embedder(cfg["embeddings"]["model"], device, cfg["embeddings"]["window"])
    with tempfile.TemporaryDirectory() as tmp:
        for pdb, csv_path in pairs:
            name = Path(pdb).stem
            try:
                if mapping:
                    pdb, csv_path = remap_chains(pdb, csv_path, mapping, tmp)
                rec = {"record_id": f"new:{name}", "complex_id": name, "pdb_code": name[:4].lower(), "set": "new",
                       "label": 0, "dockq": 0.0, "pdb_path": str(pdb), "csv_path": str(csv_path)}
                d, _ = build_graph(rec)
                g = slim_graph(d)
                if embedder:
                    g["plm"] = embedder.nodes(d["chain_seqs"], d["role"], d["seq_pos"])
                graphs.append(g); ids.append(name)
            except Exception as e:
                failed[name] = f"{type(e).__name__}: {e}"
    return graphs, ids, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("model")
    src.add_argument("--run", help="trained run name (uses runs/<run>/final_model.pt)")
    src.add_argument("--model", help="path to a final_model.pt instead of --run")
    inp = ap.add_argument_group("input (choose one)")
    inp.add_argument("--split", choices=["test", "all", "0", "1", "2", "3", "4"])
    inp.add_argument("--records", help="comma-separated record ids, e.g. exp:1a2y_1,dbl:1n8z_1")
    inp.add_argument("--pdb"); inp.add_argument("--csv")
    inp.add_argument("--input-dir", help="folder of <name>.pdb + <name>.csv pairs")
    mp = ap.add_argument_group("chain mapping for new complexes")
    mp.add_argument("--heavy"); mp.add_argument("--light"); mp.add_argument("--antigen")
    ap.add_argument("--out", help="output CSV path")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    if not (a.run or a.model):
        ap.error("give --run or --model")
    model_path = Path(a.model) if a.model else run_dir(a.run) / "final_model.pt"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found. Has training finished? Check: python -m abset_ppi.status --run {a.run}")
    device = pick_device(a.device)
    cfg, calib, models = load_final(model_path, device)
    print(f"Loaded {len(models)} fold models from {model_path} (prior {calib['prior']}, threshold {calib['threshold']:.3f})")

    labels, failed = None, {}
    if a.split or a.records:
        processed = resolve(cfg["data"]["processed_dir"])
        recs = read_records(processed)
        if a.records:
            want = set(a.records.split(","))
            recs = [r for r in recs if r["record_id"] in want]
            missing = want - {r["record_id"] for r in recs}
            if missing:
                print(f"Unknown record ids: {sorted(missing)}")
        elif a.split != "all":
            recs = [r for r in recs if r["split"] == a.split]
        graphs = []
        for r in recs:
            g = slim_graph(torch.load(processed / r["graph"], weights_only=False))
            if cfg["embeddings"]["use"]:
                p = embedding_path(resolve(cfg["data"]["embeddings_dir"]), cfg["embeddings"]["model"], r["record_id"])
                if not p.exists():
                    raise SystemExit(f"Missing embedding {p}. Run: python -m abset_ppi.embed")
                g["plm"] = torch.load(p, weights_only=True)
            graphs.append(g)
        ids = [r["record_id"] for r in recs]
        labels = [r["label"] for r in recs]
    elif a.pdb or a.input_dir:
        if a.pdb:
            if not a.csv:
                ap.error("--pdb needs --csv (AbSet-format descriptor file)")
            pairs = [(a.pdb, a.csv)]
        else:
            d = Path(a.input_dir)
            pairs = [(p, p.with_suffix(".csv")) for p in sorted(d.glob("*.pdb")) if p.with_suffix(".csv").exists()]
            skipped = [p.name for p in d.glob("*.pdb") if not p.with_suffix(".csv").exists()]
            if skipped:
                print(f"Skipping {len(skipped)} PDB files without a matching CSV: {skipped[:5]}")
        mapping = {}
        if a.heavy or a.light or a.antigen:
            if not (a.heavy and a.light and a.antigen):
                ap.error("give all three of --heavy, --light, --antigen")
            mapping = {a.heavy: "H", a.light: "L", a.antigen: "A"}
        graphs, ids, failed = featurise_new(pairs, mapping, cfg, device)
    else:
        ap.error("choose an input: --split, --records, --pdb/--csv or --input-dir")

    for name, err in failed.items():
        print(f"  could not score {name}: {err}")
    if not graphs:
        raise SystemExit("Nothing to score.")
    z, p, per_fold = score(graphs, cfg, calib, models, device)
    thr = calib["threshold"]
    out = Path(a.out) if a.out else (run_dir(a.run) if a.run else model_path.parent) / f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "probability", "predicted_binding", "mean_logit"] + (["label"] if labels else [])
                   + [f"logit_fold{k}" for k, _, _ in models])
        for i, rid in enumerate(ids):
            w.writerow([rid, f"{p[i]:.6f}", int(p[i] >= thr), f"{z[i]:.4f}"] + ([labels[i]] if labels else [])
                       + [f"{v:.4f}" for v in per_fold[:, i]])
    print(f"\n{'id':<28}{'probability':>12}{'binding?':>10}" + (f"{'label':>7}" if labels else ""))
    for i in np.argsort(-p)[:25]:
        print(f"{ids[i]:<28}{p[i]:>12.4f}{('yes' if p[i] >= thr else 'no'):>10}" + (f"{labels[i]:>7}" if labels else ""))
    if len(ids) > 25:
        print(f"... {len(ids) - 25} more")
    print(f"\nWrote {len(ids)} predictions to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
