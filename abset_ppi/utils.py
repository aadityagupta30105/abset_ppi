"""Shared helpers: paths, config, logging, atomic writes, clean stopping."""
import copy
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent          # package root folder
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
RUNS_DIR = ROOT / "runs"


# ----------------------------------------------------------------------------- config

def load_config(path=None, overrides=()):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides):
    for item in overrides or ():
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_dir(name):
    return RUNS_DIR / name


# ----------------------------------------------------------------------------- logging

def get_logger(log_file=None, name="abset_ppi"):
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def fmt_duration(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ----------------------------------------------------------------------------- files

def atomic_torch_save(obj, path):
    """Write to a temp file then rename, so an interrupted save never corrupts the previous checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return copy.deepcopy(default)


# ----------------------------------------------------------------------------- stopping

class StopFlag:
    """Clean-stop requests: Ctrl+C once, or a STOP file in the run folder.
    The trainer checks this between steps, saves, and exits. A second Ctrl+C exits immediately
    (the last saved checkpoint is still valid because saves are atomic)."""

    def __init__(self, stop_file):
        self.stop_file = Path(stop_file)
        self.requested = False
        self._hits = 0
        signal.signal(signal.SIGINT, self._handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self._hits += 1
        if self._hits >= 2:
            print("\nSecond interrupt: exiting now. The last saved checkpoint is intact.", flush=True)
            os._exit(130)
        self.requested = True
        print("\nStop requested: finishing the current step, saving, then exiting "
              "(press Ctrl+C again to exit immediately).", flush=True)

    def check(self):
        if self.stop_file.exists():
            self.requested = True
        return self.requested

    def clear(self):
        if self.stop_file.exists():
            self.stop_file.unlink()


# ----------------------------------------------------------------------------- rng

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError:
            pass                                       # different GPU count: continue with fresh CUDA RNG


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
