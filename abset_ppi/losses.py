"""Training losses: main label loss plus three auxiliary terms."""
from collections import defaultdict

import torch
import torch.nn.functional as F


def compute_losses(out, b, cfg):
    t = cfg["train"]
    w = t["loss_weights"]
    ls = t["label_smoothing"]
    target = b.y * (1 - ls) + ls / 2
    main = F.binary_cross_entropy_with_logits(out["logit"].float(), target)

    dockq = F.mse_loss(torch.sigmoid(out["dockq"].float()), b.dockq)

    known = b.native_iface >= 0
    residue = (F.binary_cross_entropy_with_logits(out["res_logit"][known].float(), b.native_iface[known])
               if known.any() else out["logit"].sum() * 0)

    # native-over-decoy margin for complexes that have both a crystal and a docked pose in the batch
    by_cx = defaultdict(lambda: [[], []])
    for i, (cx, y) in enumerate(zip(b.complex_ids, b.y.tolist())):
        by_cx[cx][int(y > 0.5)].append(i)
    terms = [F.relu(1 - (out["logit"][p] - out["logit"][n])) for negs, poss in by_cx.values() for p in poss for n in negs]
    rank = torch.stack(terms).mean() if terms else out["logit"].sum() * 0

    total = w["main"] * main + w["dockq"] * dockq + w["residue"] * residue + w["rank"] * rank
    item = lambda v: float(v.detach())
    return total, {"loss": item(total), "main": item(main), "dockq": item(dockq),
                   "residue": item(residue), "rank": item(rank), "n_rank_pairs": len(terms)}
