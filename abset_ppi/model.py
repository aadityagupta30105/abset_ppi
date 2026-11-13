"""Structure-aware graph network that scores one antibody–antigen orientation.

Pipeline: node / edge encoders -> geometric graph-transformer blocks over residue neighbours ->
antibody <-> antigen cross-attention -> gated pooling of each side + global interface features ->
logit (main), DockQ estimate and per-residue native-interface logits (auxiliary heads).
All message passing is plain PyTorch (index_add / scatter_reduce), so no graph library is needed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_EDGE_SCALAR
from .featurize import NODE_COLS, GLOBAL_COLS, REGIONS


def scatter_softmax(logits, index, num):
    """Softmax of `logits` (E, ...) over entries that share the same `index` (E,)."""
    shape = (num,) + logits.shape[1:]
    idx = index.view(-1, *([1] * (logits.dim() - 1))).expand_as(logits)
    mx = torch.full(shape, float("-inf"), device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(0, idx, logits, reduce="amax", include_self=True)
    ex = (logits - mx[index]).exp()
    den = torch.zeros(shape, device=logits.device, dtype=logits.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


class RBF(nn.Module):
    def __init__(self, bins, d_max):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, d_max, bins))
        self.gamma = (bins / d_max) ** 2

    def forward(self, d):                                 # (E, k) -> (E, k*bins)
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2).flatten(1)


def mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class EdgeAttention(nn.Module):
    """Multi-head attention of each receiver over its incoming edges, biased by edge features."""

    def __init__(self, hidden, edge_dim, heads, dropout):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.dh = heads, hidden // heads
        self.q, self.k, self.v = (nn.Linear(hidden, hidden) for _ in range(3))
        self.ev = nn.Linear(edge_dim, hidden)
        self.eb = nn.Linear(edge_dim, heads)
        self.o = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, src, dst, extra_bias=None):
        E, N = src.shape[0], h.shape[0]
        q = self.q(h)[dst].view(E, self.h, self.dh)
        k = self.k(h)[src].view(E, self.h, self.dh)
        v = (self.v(h)[src] + self.ev(e)).view(E, self.h, self.dh)
        logits = ((q * k).sum(-1) / math.sqrt(self.dh) + self.eb(e)).float()
        if extra_bias is not None:
            logits = logits + extra_bias.float()
        a = scatter_softmax(logits, dst, N).to(v.dtype)
        out = torch.zeros(N, self.h, self.dh, device=h.device, dtype=v.dtype).index_add_(0, dst, a.unsqueeze(-1) * v)
        return self.drop(self.o(out.reshape(N, -1)))


class GraphBlock(nn.Module):
    def __init__(self, hidden, edge_hidden, heads, dropout):
        super().__init__()
        self.att = EdgeAttention(hidden, edge_hidden, heads, dropout)
        self.n1, self.n2, self.ne = nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(edge_hidden)
        self.ff = mlp(hidden, 2 * hidden, hidden, dropout)
        self.eup = mlp(2 * hidden + edge_hidden, edge_hidden, edge_hidden, dropout)

    def forward(self, h, e, src, dst):
        h = self.n1(h + self.att(h, e, src, dst))
        h = self.n2(h + self.ff(h))
        e = self.ne(e + self.eup(torch.cat([h[src], h[dst], e], -1)))
        return h, e


class CrossBlock(nn.Module):
    """Antibody interface residues attend to antigen interface residues and vice versa."""

    def __init__(self, hidden, heads, dropout, rbf_bins):
        super().__init__()
        self.rbf = RBF(rbf_bins, 32.0)
        self.att = EdgeAttention(hidden, rbf_bins, heads, dropout)
        self.lam = nn.Parameter(torch.zeros(heads))       # learned distance decay per head
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = mlp(hidden, 2 * hidden, hidden, dropout)

    def forward(self, h, src, dst, dist):
        e = self.rbf(dist.unsqueeze(1))
        bias = -F.softplus(self.lam).unsqueeze(0) * dist.unsqueeze(1) / 4.0
        h = self.n1(h + self.att(h, e, src, dst, extra_bias=bias))
        return self.n2(h + self.ff(h))


class PoseScorer(nn.Module):
    def __init__(self, cfg, use_plm, plm_dim):
        super().__init__()
        m = cfg["model"]
        H, EH, heads, dr, bins = m["hidden"], m["edge_hidden"], m["heads"], m["dropout"], m["rbf_bins"]
        self.use_plm = use_plm
        self.aa = nn.Embedding(21, 16)
        self.ss = nn.Embedding(10, 8)
        self.role = nn.Embedding(3, 4)
        self.region = nn.Embedding(len(REGIONS), 8)
        node_in = len(NODE_COLS) + 16 + 8 + 4 + 8
        if use_plm:
            self.plm = nn.Sequential(nn.Dropout(dr), nn.Linear(plm_dim, m["plm_proj"]), nn.GELU())
            node_in += m["plm_proj"]
        self.node_enc = nn.Sequential(nn.Linear(node_in, H), nn.GELU(), nn.LayerNorm(H))
        self.rbf = RBF(bins, 20.0)
        self.etype = nn.Embedding(3, 8)
        self.edge_enc = nn.Sequential(nn.Linear(4 * bins + N_EDGE_SCALAR + 8, EH), nn.GELU(), nn.LayerNorm(EH))
        self.blocks = nn.ModuleList(GraphBlock(H, EH, heads, dr) for _ in range(m["graph_layers"]))
        self.cross = nn.ModuleList(CrossBlock(H, heads, dr, bins) for _ in range(m["cross_layers"]))
        self.gate = nn.Linear(H, 1)
        self.glob = mlp(len(GLOBAL_COLS), 64, 64, dr)
        self.head = mlp(2 * H + 64, 128, 1, dr)
        self.dockq_head = mlp(2 * H + 64, 64, 1, dr)
        self.res_head = nn.Linear(H, 1)

    def forward(self, b):
        parts = [b.x, self.aa(b.aa), self.ss(b.ss), self.role(b.role), self.region(b.region)]
        if self.use_plm:
            parts.append(self.plm(b.plm))
        h = self.node_enc(torch.cat(parts, -1))
        e = self.edge_enc(torch.cat([self.rbf(b.edge_dist), b.edge_scal, self.etype(b.edge_type)], -1))
        src, dst = b.edge_index
        for blk in self.blocks:
            h, e = blk(h, e, src, dst)
        cs, cd = b.cross_index
        for blk in self.cross:
            h = blk(h, cs, cd, b.cross_dist)
        # gated attention pooling, separately over antibody and antigen nodes of each graph
        side = (b.role == 2).long()
        group = b.node_graph * 2 + side
        w = scatter_softmax(self.gate(h).squeeze(-1).float(), group, 2 * b.num_graphs).to(h.dtype)
        pooled = torch.zeros(2 * b.num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        pooled = pooled.index_add_(0, group, w.unsqueeze(-1) * h).view(b.num_graphs, -1)
        z = torch.cat([pooled, self.glob(b.glob)], -1)
        return {"logit": self.head(z).squeeze(-1), "dockq": self.dockq_head(z).squeeze(-1),
                "res_logit": self.res_head(h).squeeze(-1)}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
