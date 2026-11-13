"""Compute protein language-model embeddings for every graph node (run once before training).

    python -m abset_ppi.embed                       # model from config/default.yaml (ESM-2 650M)
    python -m abset_ppi.embed --model facebook/esm2_t30_150M_UR50D   # smaller, faster (dim 640)

Each chain (H, L, antigen) is embedded on its full sequence; the vectors of the residues that are graph
nodes are saved as data/embeddings/<model>/<record>.pt (float16, nodes x dim). The job is resumable:
records that already have a file are skipped, so Ctrl+C and re-running is safe.
The model weights are downloaded from Hugging Face on first use (650M model: ~2.6 GB).
"""
import argparse
import csv
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

from .data import embedding_path, read_records
from .utils import atomic_torch_save, fmt_duration, get_logger, load_config, pick_device, resolve

ROLES = ("H", "L", "AG")


class Embedder:
    def __init__(self, model_id, device, window=1000, cache_size=4000):
        from transformers import AutoTokenizer, EsmModel
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = EsmModel.from_pretrained(model_id, add_pooling_layer=False).to(device).eval()
        if device.type == "cuda":
            self.model.half()
        self.device, self.window = device, window
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def _run(self, seq):
        enc = self.tok(seq, return_tensors="pt", add_special_tokens=True).to(self.device)
        h = self.model(**enc).last_hidden_state[0, 1:-1].float().cpu()      # drop BOS / EOS
        assert h.shape[0] == len(seq), (h.shape, len(seq))
        return h

    def sequence(self, seq):
        if seq in self.cache:
            self.cache.move_to_end(seq)
            return self.cache[seq]
        if len(seq) <= self.window:
            h = self._run(seq)
        else:                                    # overlapping windows, averaged where they overlap
            step = self.window - 200
            acc = torch.zeros(len(seq), self.dim); cnt = torch.zeros(len(seq), 1)
            for s in range(0, len(seq), step):
                e = min(len(seq), s + self.window)
                acc[s:e] += self._run(seq[s:e]); cnt[s:e] += 1
                if e == len(seq):
                    break
            h = acc / cnt
        self.cache[seq] = h
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return h

    def nodes(self, chain_seqs, role, seq_pos):
        out = torch.zeros(len(role), self.dim)
        for r_idx, r_name in enumerate(ROLES):
            m = role == r_idx
            if m.any():
                out[m] = self.sequence(chain_seqs[r_name])[seq_pos[m]]
        return out.half()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="Hugging Face model id (overrides the config)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only the first N records (testing)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    model_id = a.model or cfg["embeddings"]["model"]
    processed = resolve(cfg["data"]["processed_dir"])
    emb_root = resolve(cfg["data"]["embeddings_dir"])
    emb_root.mkdir(parents=True, exist_ok=True)
    log = get_logger(emb_root / "embed.log")
    device = pick_device(a.device)

    records = read_records(processed)
    if a.limit:
        records = records[:a.limit]
    seqs = {}
    with open(processed / "sequences.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.setdefault(r["record_id"], {})[r["chain_role"]] = r["sequence"]
    todo = [r for r in records if not embedding_path(emb_root, model_id, r["record_id"]).exists()]
    log.info(f"Embeddings with {model_id} on {device}: {len(records) - len(todo)} done, {len(todo)} to go.")
    if not todo:
        log.info("Nothing to do.")
        return 0
    emb = Embedder(model_id, device, cfg["embeddings"]["window"])
    if emb.dim != cfg["embeddings"]["dim"] or model_id != cfg["embeddings"]["model"]:
        log.info(f"NOTE: when training, pass  --set embeddings.model={model_id} embeddings.dim={emb.dim}")
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        g = torch.load(processed / r["graph"], weights_only=False)
        vec = emb.nodes(seqs[r["record_id"]], g["role"], g["seq_pos"])
        out = embedding_path(emb_root, model_id, r["record_id"])
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(vec, out)
        if i % 100 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            log.info(f"  {i}/{len(todo)} records | {fmt_duration(time.time() - t0)} elapsed | "
                     f"~{fmt_duration(rate * (len(todo) - i))} left")
    log.info(f"Done. Embeddings are in {embedding_path(emb_root, model_id, 'x:y').parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
