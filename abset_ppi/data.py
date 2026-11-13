"""Loading cached graphs, fold normalisation and batching."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .featurize import NODE_COLS, EDGE_COLS, GLOBAL_COLS

# Node columns that are clipped to the training 0.5–99.5 percentile range and z-scored.
# Masks, the interface flag and sin/cos torsions are already on a fixed scale and pass through unchanged.
NODE_PASS = {"phi_sin", "phi_cos", "psi_sin", "psi_cos", "interface", "m_hseB", "m_hseA", "m_ss", "m_tors"}
NODE_SCALE = [c for c in NODE_COLS if c not in NODE_PASS]
I_MIN_DIST = NODE_COLS.index("min_dist_partner")
E_DIST = [EDGE_COLS.index(c) for c in ("d_CA", "d_CB", "d_NO", "d_ON")]
E_GEOM = [EDGE_COLS.index(c) for c in ("q_w", "q_x", "q_y", "q_z", "dir_x", "dir_y", "dir_z", "same_chain", "salt_bridge")]
E_SEQ = EDGE_COLS.index("seq_sep")
E_LOG = [EDGE_COLS.index(c) for c in ("n_contacts", "n_hbond", "n_hydrophobic")]
N_EDGE_SCALAR = len(E_GEOM) + 1 + len(E_LOG)          # edge scalars passed to the model besides distances

KEEP_KEYS = ("x", "aa", "ss", "role", "region", "seq_pos", "edge_index", "edge_type", "edge_attr",
             "global_feats", "y", "dockq", "native_iface")


def model_slug(model_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.split("/")[-1])


def embedding_path(emb_root, model_id, record_id):
    return Path(emb_root) / model_slug(model_id) / (record_id.replace(":", "__") + ".pt")


def read_records(processed_dir):
    with open(Path(processed_dir) / "records.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("graph")]
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def slim_graph(d):
    """Keep only what training needs; CA coordinates for cross-attention distances."""
    g = {k: d[k] for k in KEEP_KEYS}
    g["ca"] = d["coords"][:, 1].float()
    g["record_id"], g["complex_id"] = d["record_id"], d["complex_id"]
    return g


class GraphStore:
    """All graphs (and optional embeddings) held in CPU memory, keyed by record_id."""

    def __init__(self, processed_dir, records, emb_root=None, emb_model=None, log=None):
        self.graphs = {}
        processed_dir = Path(processed_dir)
        missing_emb = []
        for k, r in enumerate(records, 1):
            d = torch.load(processed_dir / r["graph"], weights_only=False)
            g = slim_graph(d)
            if emb_model:
                p = embedding_path(emb_root, emb_model, r["record_id"])
                if p.exists():
                    g["plm"] = torch.load(p, weights_only=True)
                else:
                    missing_emb.append(r["record_id"])
            self.graphs[r["record_id"]] = g
            if log and k % 1000 == 0:
                log.info(f"  loaded {k}/{len(records)} graphs")
        if missing_emb:
            raise SystemExit(
                f"{len(missing_emb)} records have no language-model embedding (first: {missing_emb[0]}).\n"
                "Run `python -m abset_ppi.embed` first, or train without embeddings with "
                "`--set embeddings.use=false`.")

    def __getitem__(self, rid):
        return self.graphs[rid]


class Normalizer:
    """Fold-specific scaling built from norm_stats.json (training records of that fold only)."""

    def __init__(self, stats):
        def vecs(block, cols):
            lo = torch.tensor([block[c]["p0.5"] for c in cols])
            hi = torch.tensor([block[c]["p99.5"] for c in cols])
            mu = torch.tensor([block[c]["mean"] for c in cols])
            sd = torch.tensor([block[c]["std"] for c in cols])
            return lo, hi, mu, sd
        self.node_idx = torch.tensor([NODE_COLS.index(c) for c in NODE_SCALE])
        self.node = vecs(stats["node"], NODE_SCALE)
        self.glob = vecs(stats["global"], GLOBAL_COLS)
        self.stats = stats

    @staticmethod
    def _scale(v, p):
        lo, hi, mu, sd = (t.to(v.device) for t in p)
        return (torch.minimum(torch.maximum(v, lo), hi) - mu) / sd

    def nodes(self, x):
        x = x.clone()
        idx = self.node_idx.to(x.device)
        x[:, idx] = self._scale(x[:, idx], self.node)
        return x

    def globals(self, g):
        return self._scale(g, self.glob)

    @staticmethod
    def edges(e):
        """Returns (distances (E,4) in Å, scalars (E, N_EDGE_SCALAR))."""
        dist = e[:, E_DIST]
        seq = (e[:, E_SEQ] / 32.0).unsqueeze(1)
        scal = torch.cat([e[:, E_GEOM], seq, torch.log1p(e[:, E_LOG].clamp(min=0))], 1)
        return dist, scal


@dataclass
class Batch:
    x: torch.Tensor
    aa: torch.Tensor
    ss: torch.Tensor
    role: torch.Tensor
    region: torch.Tensor
    min_dist: torch.Tensor
    plm: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_dist: torch.Tensor
    edge_scal: torch.Tensor
    cross_index: torch.Tensor
    cross_dist: torch.Tensor
    node_graph: torch.Tensor
    glob: torch.Tensor
    y: torch.Tensor
    dockq: torch.Tensor
    native_iface: torch.Tensor
    num_graphs: int
    record_ids: list
    complex_ids: list

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device, non_blocking=True))
        return self


def collate(graphs, normalizer, cross_cutoff, use_plm):
    xs, edge_idx, cross_idx, cross_d = [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        n = g["x"].shape[0]
        edge_idx.append(g["edge_index"] + offset)
        md = g["x"][:, I_MIN_DIST]
        ab = torch.where((g["role"] < 2) & (md <= cross_cutoff))[0]
        ag = torch.where((g["role"] == 2) & (md <= cross_cutoff))[0]
        if len(ab) == 0 or len(ag) == 0:                 # never expected (every graph has an interface)
            ab = torch.where(g["role"] < 2)[0][md[g["role"] < 2].argsort()[:8]]
            ag = torch.where(g["role"] == 2)[0][md[g["role"] == 2].argsort()[:8]]
        s = ab.repeat_interleave(len(ag)); t = ag.repeat(len(ab))
        pairs = torch.cat([torch.stack([s, t]), torch.stack([t, s])], 1)      # both directions
        cross_idx.append(pairs + offset)
        cross_d.append((g["ca"][pairs[0]] - g["ca"][pairs[1]]).norm(dim=1))
        xs.append(torch.full((n,), gi, dtype=torch.long))
        offset += n
    cat = lambda k: torch.cat([g[k] for g in graphs])
    x_raw = cat("x")
    e_raw = cat("edge_attr")
    edge_dist, edge_scal = normalizer.edges(e_raw)
    plm = cat("plm").float() if use_plm else torch.zeros(0)
    return Batch(
        x=normalizer.nodes(x_raw), aa=cat("aa"), ss=cat("ss"), role=cat("role"), region=cat("region"),
        min_dist=x_raw[:, I_MIN_DIST], plm=plm,
        edge_index=torch.cat(edge_idx, 1), edge_type=cat("edge_type"), edge_dist=edge_dist, edge_scal=edge_scal,
        cross_index=torch.cat(cross_idx, 1), cross_dist=torch.cat(cross_d),
        node_graph=torch.cat(xs), glob=normalizer.globals(torch.stack([g["global_feats"] for g in graphs])),
        y=torch.stack([g["y"] for g in graphs]).float(), dockq=torch.stack([g["dockq"] for g in graphs]).float(),
        native_iface=cat("native_iface").float(), num_graphs=len(graphs),
        record_ids=[g["record_id"] for g in graphs], complex_ids=[g["complex_id"] for g in graphs])


def load_norm_stats(processed_dir, key):
    with open(Path(processed_dir) / "norm_stats.json", encoding="utf-8") as f:
        return json.load(f)[key]
