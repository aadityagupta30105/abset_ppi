"""Balanced resampling: every epoch uses all training negatives and an equal number of positives.

Positives come from a shuffled queue that refills when empty, so over ~10 epochs every positive is seen.
With probability `matched_pair_prob`, a negative brings its crystal counterpart (same complex) into the
epoch and into the same batch, which feeds the native-over-decoy ranking loss.
The sampler state (queue and RNG) is saved in checkpoints so a resumed run continues the same sequence.
"""
import random


class BalancedResampler:
    def __init__(self, pos_ids, neg_ids, matched, batch_size, matched_pair_prob, seed):
        self.pos, self.neg = sorted(pos_ids), sorted(neg_ids)
        self.matched = {n: p for n, p in matched.items() if n in set(self.neg) and p in set(self.pos)}
        self.bs, self.p_pair = batch_size, matched_pair_prob
        self.rng = random.Random(seed)
        self.queue = []

    def _take_positive(self, exclude):
        while True:
            if not self.queue:
                q = self.pos[:]
                self.rng.shuffle(q)
                self.queue = q
            p = self.queue.pop()
            if p not in exclude:
                return p

    def epoch_batches(self):
        units, chosen = [], set()
        for n in self.neg:
            p = self.matched.get(n)
            if p is not None and p not in chosen and self.rng.random() < self.p_pair:
                units.append([n, p]); chosen.add(p)
            else:
                units.append([n])
        target = min(len(self.neg), len(self.pos))   # each positive at most once per epoch
        while len(chosen) < target:
            p = self._take_positive(chosen)
            units.append([p]); chosen.add(p)
        self.rng.shuffle(units)
        flat = [r for u in units for r in u]
        return [flat[i:i + self.bs] for i in range(0, len(flat), self.bs)]

    def state_dict(self):
        return {"queue": list(self.queue), "rng": self.rng.getstate()}

    def load_state_dict(self, st):
        self.queue = list(st["queue"])
        rng = st["rng"]
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
