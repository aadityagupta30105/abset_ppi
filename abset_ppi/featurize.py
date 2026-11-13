"""Featurisation: turns one antibody–antigen complex (PDB + AbSet-format descriptor CSV) into a graph.

This is the exact code that produced data/processed/. It is used again at prediction time so that new
complexes get identical features. Running this module as a script rebuilds the whole processed dataset
from an AbSet_filtered folder (not needed for training; kept for reproducibility):

    python -m abset_ppi.featurize --data AbSet_filtered --out data/processed
"""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree

SETS = [("exp", "Experimental", 1), ("dsd", "Docking_MonomersFromModeling", 0), ("dbl", "Docking_MonomersFromXtal", 0)]

AA3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
       "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA3)}          # 20 = unknown
SS_CODES = "HGIEBTSPL"                               # 9 = missing
# Theoretical max ASA, Tien et al. 2013 (Å^2)
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
           "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
           "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}

ROLE = {"H": 0, "L": 1}                               # anything else = antigen (2)
REGIONS = ["H-FR1", "H-CDR1", "H-FR2", "H-CDR2", "H-FR3", "H-CDR3", "H-FR4",
           "L-FR1", "L-CDR1", "L-FR2", "L-CDR2", "L-FR3", "L-CDR3", "L-FR4", "Ab-other", "Antigen"]
# Chothia CDR definitions on Martin / Chothia numbering (inclusive, insertion codes ride on their number)
CDR = {"H": [(26, 32), (52, 56), (95, 102)], "L": [(24, 34), (50, 56), (89, 97)]}
FV_END = {"H": 113, "L": 107}

CATION = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
ANION = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
APOLAR_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "TYR"}
BACKBONE = {"N", "CA", "C", "O"}

IFACE_CUT, HB_CUT, SB_CUT = 4.5, 3.5, 4.0
CROP_CUT, NEIGH_CUT, MAX_NODES = 14.0, 8.0, 320
KNN, KNN_RADIUS, SEQ_K = 24, 15.0, 2
MIN_IFACE_RES = 10                                    # review rule: reject smaller protein interfaces

NODE_COLS = ["SASA", "RASA", "HSE_B_U", "HSE_B_D", "HSE_A_U", "HSE_A_D", "mean_IP", "max_IP", "min_IP",
             "mean_dpx", "phi_sin", "phi_cos", "psi_sin", "psi_cos",
             "sasa_cplx", "sasa_unb", "rasa_unb", "dsasa", "drasa", "min_dist_partner", "interface",
             "m_hseB", "m_hseA", "m_ss", "m_tors"]
EDGE_COLS = ["d_CA", "d_CB", "d_NO", "d_ON", "q_w", "q_x", "q_y", "q_z", "dir_x", "dir_y", "dir_z",
             "seq_sep", "same_chain", "n_contacts", "n_hbond", "salt_bridge", "n_hydrophobic"]
EDGE_TYPES = ["seq", "intra", "inter"]
GLOBAL_COLS = ["n_ab_iface", "n_ag_iface", "bsa_total", "bsa_polar", "bsa_apolar", "frac_paratope_cdr",
               "frac_paratope_h3", "n_hbond", "n_salt_bridge", "n_hydrophobic_pairs", "contact_density"]


# ----------------------------------------------------------------------------- helpers

def region_of(role_chain, num, is_ab):
    if not is_ab:
        return REGIONS.index("Antigen")
    c = role_chain
    if num > FV_END[c]:
        return REGIONS.index("Ab-other")
    (a1, b1), (a2, b2), (a3, b3) = CDR[c]
    order = [(a1, "FR1"), (b1 + 1, "FR2"), (a2, "CDR2"), (b2 + 1, "FR3"), (a3, "CDR3"), (b3 + 1, "FR4")]
    name = "FR1" if num < a1 else "CDR1"
    for start, nm in order[1:]:
        if num >= start:
            name = nm
    return REGIONS.index(f"{c}-{name}")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def frames(N, CA, C):
    x = C - CA
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    y = N - CA
    y -= (y * x).sum(1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=2)               # (n, 3, 3), columns = axes


def rot_to_quat(R):
    """R: (m,3,3) -> (m,4) unit quaternion (w,x,y,z), w >= 0."""
    m = R.shape[0]
    q = np.zeros((m, 4))
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = np.sqrt(np.clip(1 + t, 0, None)) / 2
    x = np.sqrt(np.clip(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], 0, None)) / 2
    y = np.sqrt(np.clip(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], 0, None)) / 2
    z = np.sqrt(np.clip(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2], 0, None)) / 2
    x = np.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = np.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = np.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q[:] = np.stack([w, x, y, z], 1)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)


def akey(a):
    res = a.get_parent()
    return (res.get_parent().id, res.id, a.get_id())


def read_csv_desc(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["chain"], r["residue_num"].strip())] = r
    return out


# ----------------------------------------------------------------------------- per record

def build_graph(rec):
    """rec needs: record_id, complex_id, pdb_code, set, label, dockq, pdb_path, csv_path.
    Returns (graph dict, info dict). Raises ValueError for complexes that break the dataset rules."""
    rid = rec["record_id"]
    structure = PDBParser(QUIET=True).get_structure(rid, rec["pdb_path"])
    model = next(iter(structure))
    # protein heavy atoms only
    for chain in list(model):
        for res in list(chain):
            if res.id[0] != " ":
                chain.detach_child(res.id)
                continue
            for atom in list(res):
                if atom.element in ("H", "D") or atom.get_id().startswith("H"):
                    res.detach_child(atom.get_id())
            if len(res) == 0:
                chain.detach_child(res.id)
        if len(chain) == 0:
            model.detach_child(chain.id)
    chain_ids = [c.id for c in model]
    ag_chains = [c for c in chain_ids if c not in ROLE]
    if not ({"H", "L"} <= set(chain_ids)) or len(ag_chains) != 1:
        raise ValueError(f"unexpected chains {chain_ids}")
    ag = ag_chains[0]

    sr = ShrakeRupley()
    sr.compute(model, level="A")
    cplx_sasa = {akey(a): a.sasa for a in model.get_atoms()}
    ab_only = model.copy(); ab_only.detach_child(ag)
    sr.compute(ab_only, level="A")
    ag_only = model.copy(); ag_only.detach_child("H"); ag_only.detach_child("L")
    sr.compute(ag_only, level="A")
    unb_sasa = {akey(a): a.sasa for a in list(ab_only.get_atoms()) + list(ag_only.get_atoms())}

    desc = read_csv_desc(rec["csv_path"])

    # residues
    residues, atoms_xyz, atoms_res, atoms_info = [], [], [], []
    for chain in model:
        for res in chain:
            ri = len(residues)
            residues.append((chain.id, res))
            for a in res:
                atoms_xyz.append(a.coord)
                atoms_res.append(ri)
                atoms_info.append((res.get_resname(), a.get_id(), a.element, akey(a)))
    atoms_xyz = np.asarray(atoms_xyz, dtype=np.float64)
    atoms_res = np.asarray(atoms_res)
    is_ab_atom = np.array([residues[r][0] in ROLE for r in atoms_res])

    # interface contacts (heavy atoms)
    ab_idx, ag_idx = np.where(is_ab_atom)[0], np.where(~is_ab_atom)[0]
    t_ab, t_ag = cKDTree(atoms_xyz[ab_idx]), cKDTree(atoms_xyz[ag_idx])
    pairs = t_ab.query_ball_tree(t_ag, IFACE_CUT)
    pair_stats = defaultdict(lambda: [0, 0, 0, 0])     # (res_ab, res_ag) -> contacts, hb, sb, hydrophobic
    n_hb_atoms = 0
    for i_local, js in enumerate(pairs):
        ia = ab_idx[i_local]
        ra = atoms_res[ia]
        rn_a, an_a, el_a, _ = atoms_info[ia]
        for j_local in js:
            ja = ag_idx[j_local]
            rb = atoms_res[ja]
            rn_b, an_b, el_b, _ = atoms_info[ja]
            st = pair_stats[(ra, rb)]
            st[0] += 1
            d = np.linalg.norm(atoms_xyz[ia] - atoms_xyz[ja])
            if el_a in ("N", "O") and el_b in ("N", "O") and d <= HB_CUT:
                st[1] += 1; n_hb_atoms += 1
            if d <= SB_CUT and (((rn_a, an_a) in CATION and (rn_b, an_b) in ANION) or
                                ((rn_a, an_a) in ANION and (rn_b, an_b) in CATION)):
                st[2] = 1
            if (el_a == "C" and el_b == "C" and an_a not in BACKBONE and an_b not in BACKBONE
                    and rn_a in APOLAR_RES and rn_b in APOLAR_RES):
                st[3] += 1
    iface_res = {r for p in pair_stats for r in p}

    # min heavy-atom distance from each residue to the partner protein
    d_to_ag, _ = t_ag.query(atoms_xyz[ab_idx])
    d_to_ab, _ = t_ab.query(atoms_xyz[ag_idx])
    min_dist = np.full(len(residues), np.inf)
    np.minimum.at(min_dist, atoms_res[ab_idx], d_to_ag)
    np.minimum.at(min_dist, atoms_res[ag_idx], d_to_ab)

    # per-residue SASA (complex / unbound), buried SASA split polar / apolar
    res_sasa_c = np.zeros(len(residues)); res_sasa_u = np.zeros(len(residues))
    bsa_polar = bsa_apolar = 0.0
    for k, (rn, an, el, key) in enumerate(atoms_info):
        c, u = cplx_sasa[key], unb_sasa[key]
        res_sasa_c[atoms_res[k]] += c; res_sasa_u[atoms_res[k]] += u
        if el in ("N", "O"):
            bsa_polar += u - c
        else:
            bsa_apolar += u - c

    # chain sequences (residues with CA) and per-residue sequence position
    chain_seq, seq_pos = defaultdict(list), {}
    for ri, (cid, res) in enumerate(residues):
        if "CA" in res:
            seq_pos[ri] = len(chain_seq[cid])
            chain_seq[cid].append(AA1[AA_IDX[res.get_resname()]] if res.get_resname() in AA_IDX else "X")

    # node selection: residues with N, CA, C near the interface, plus their CA neighbours
    has_bb = np.array([all(n in res for n in ("N", "CA", "C")) for _, res in residues])
    cand = np.where(has_bb)[0]
    core = cand[min_dist[cand] <= CROP_CUT]
    ca_all = np.array([residues[i][1]["CA"].coord for i in cand])
    tree_ca = cKDTree(ca_all)
    near = set(core.tolist())
    for lst in tree_ca.query_ball_point(np.array([residues[i][1]["CA"].coord for i in core]), NEIGH_CUT):
        near.update(cand[lst].tolist())
    nodes = sorted(near, key=lambda i: (min_dist[i], i))[:MAX_NODES]
    nodes = sorted(nodes)                                  # chain order
    n = len(nodes)
    if n == 0:
        raise ValueError("no interface nodes (antibody and antigen not in contact)")
    if len(iface_res) < MIN_IFACE_RES:
        raise ValueError(f"only {len(iface_res)} interface residues (< {MIN_IFACE_RES})")
    node_pos = {ri: k for k, ri in enumerate(nodes)}

    # node tensors
    x = np.zeros((n, len(NODE_COLS)), dtype=np.float32)
    aa = np.zeros(n, dtype=np.int64); ss = np.zeros(n, dtype=np.int64)
    role = np.zeros(n, dtype=np.int64); region = np.zeros(n, dtype=np.int64)
    spos = np.zeros(n, dtype=np.int64)
    keys = []
    csv_missing = 0
    N_, CA_, C_, O_ = (np.zeros((n, 3)) for _ in range(4))
    for k, ri in enumerate(nodes):
        cid, res = residues[ri]
        rn = res.get_resname()
        num, icode = res.id[1], res.id[2].strip()
        key = f"{num}{icode}"
        keys.append((cid if cid in ROLE else "AG", key))
        aa[k] = AA_IDX.get(rn, 20)
        role[k] = ROLE.get(cid, 2)
        region[k] = region_of(cid, num, cid in ROLE)
        spos[k] = seq_pos[ri]
        N_[k], CA_[k], C_[k] = res["N"].coord, res["CA"].coord, res["C"].coord
        O_[k] = res["O"].coord if "O" in res else res["C"].coord
        r = desc.get((cid, key))
        v = dict.fromkeys(NODE_COLS, 0.0)
        if r is None:
            csv_missing += 1
            ss[k] = 9
        else:
            for col in ("SASA", "RASA", "mean_IP", "max_IP", "min_IP", "mean_dpx"):
                v[col] = fnum(r[col]) or 0.0
            hb = [fnum(r[c]) for c in ("EXP_HSE_B_U", "EXP_HSE_B_D")]
            ha = [fnum(r[c]) for c in ("EXP_HSE_A_U", "EXP_HSE_A_D")]
            if None in hb: v["m_hseB"] = 1.0
            else: v["HSE_B_U"], v["HSE_B_D"] = hb
            if None in ha: v["m_hseA"] = 1.0
            else: v["HSE_A_U"], v["HSE_A_D"] = ha
            s = r["secondary_structure"].strip()
            ss[k] = SS_CODES.index(s) if s in SS_CODES and s else 9
            if ss[k] == 9: v["m_ss"] = 1.0
            phi, psi = fnum(r["phi"]), fnum(r["psi"])
            if phi is None or psi is None or abs(phi) > 180 or abs(psi) > 180:
                v["m_tors"] = 1.0
            else:
                v["phi_sin"], v["phi_cos"] = math.sin(math.radians(phi)), math.cos(math.radians(phi))
                v["psi_sin"], v["psi_cos"] = math.sin(math.radians(psi)), math.cos(math.radians(psi))
        mx = MAX_ASA.get(rn, 200.0)
        v["sasa_cplx"], v["sasa_unb"] = res_sasa_c[ri], res_sasa_u[ri]
        v["rasa_unb"] = 100 * res_sasa_u[ri] / mx
        v["dsasa"] = res_sasa_u[ri] - res_sasa_c[ri]
        v["drasa"] = 100 * v["dsasa"] / mx
        v["min_dist_partner"] = min(min_dist[ri], 99.0)
        v["interface"] = float(ri in iface_res)
        x[k] = [v[c] for c in NODE_COLS]

    if set(role.tolist()) != {0, 1, 2}:
        raise ValueError("interface crop lacks one of the chains H, L, antigen")

    # virtual CB (ideal geometry), frames
    b, c = CA_ - N_, C_ - CA_
    a = np.cross(b, c)
    CB_ = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA_
    R = frames(N_, CA_, C_)

    # edges: kNN within radius + sequence neighbours
    D = np.linalg.norm(CA_[:, None] - CA_[None], axis=2)
    np.fill_diagonal(D, np.inf)
    edges = set()
    kk = min(KNN, n - 1)
    if kk > 0:
        nbr = np.argsort(D, axis=1)[:, :kk]
        for i in range(n):
            for j in nbr[i]:
                if D[i, j] <= KNN_RADIUS:
                    edges.add((int(j), i))                 # message j -> i
    chain_of = [residues[ri][0] for ri in nodes]
    for i in range(n):
        for s in range(1, SEQ_K + 1):
            j = i + s
            if j < n and chain_of[j] == chain_of[i] and spos[j] - spos[i] <= SEQ_K:
                edges.add((i, j)); edges.add((j, i))
    edges = sorted(edges)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)

    same_chain = np.array([chain_of[s] == chain_of[d] for s, d in edges], dtype=bool)
    ab_s, ab_d = role[src] < 2, role[dst] < 2
    sep = np.where(same_chain, spos[src] - spos[dst], 0)
    etype = np.where(ab_s != ab_d, 2, np.where(same_chain & (np.abs(sep) <= SEQ_K), 0, 1))

    def dist(A, B):
        return np.linalg.norm(A[dst] - B[src], axis=1)      # receiver i = dst, sender j = src
    Rrel = np.einsum("eji,ejk->eik", R[dst], R[src])         # R_i^T R_j
    vec = CA_[src] - CA_[dst]
    dirv = np.einsum("eji,ej->ei", R[dst], vec) / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    chem = np.zeros((len(edges), 4))
    for e, (s, d) in enumerate(edges):
        if etype[e] == 2:
            rs, rd = nodes[s], nodes[d]
            key = (rs, rd) if role[s] < 2 else (rd, rs)
            st = pair_stats.get(key)
            if st:
                chem[e] = st
    eattr = np.concatenate([
        np.stack([dist(CA_, CA_), dist(CB_, CB_), dist(N_, O_), dist(O_, N_)], 1),
        rot_to_quat(Rrel), dirv,
        np.clip(sep, -32, 32)[:, None], same_chain[:, None].astype(float), chem], 1).astype(np.float32)

    # global interface features
    ab_if = [r for r in iface_res if residues[r][0] in ROLE]
    ag_if = [r for r in iface_res if residues[r][0] not in ROLE]
    cdr_ids = {REGIONS.index(f"{c}-CDR{i}") for c in "HL" for i in (1, 2, 3)}
    ab_regions = [region_of(residues[r][0], residues[r][1].id[1], True) for r in ab_if]
    n_contacts = sum(s[0] for s in pair_stats.values())
    g = {
        "n_ab_iface": len(ab_if), "n_ag_iface": len(ag_if),
        "bsa_total": bsa_polar + bsa_apolar, "bsa_polar": bsa_polar, "bsa_apolar": bsa_apolar,
        "frac_paratope_cdr": (sum(r in cdr_ids for r in ab_regions) / len(ab_regions)) if ab_regions else 0.0,
        "frac_paratope_h3": (sum(r == REGIONS.index("H-CDR3") for r in ab_regions) / len(ab_regions)) if ab_regions else 0.0,
        "n_hbond": n_hb_atoms, "n_salt_bridge": sum(s[2] for s in pair_stats.values()),
        "n_hydrophobic_pairs": sum(1 for s in pair_stats.values() if s[3] > 0),
        "contact_density": n_contacts / max(1, len(iface_res)),
    }

    data = {
        "record_id": rid, "complex_id": rec["complex_id"], "pdb_code": rec["pdb_code"], "set": rec["set"],
        "y": torch.tensor(float(rec["label"])), "dockq": torch.tensor(float(rec["dockq"])),
        "x": torch.from_numpy(x), "aa": torch.from_numpy(aa), "ss": torch.from_numpy(ss),
        "role": torch.from_numpy(role), "region": torch.from_numpy(region), "seq_pos": torch.from_numpy(spos),
        "coords": torch.from_numpy(np.stack([N_, CA_, C_, O_, CB_], 1).astype(np.float32)),  # (n,5,3)
        "edge_index": torch.from_numpy(np.stack([src, dst])), "edge_type": torch.from_numpy(etype.astype(np.int64)),
        "edge_attr": torch.from_numpy(eattr),
        "global_feats": torch.tensor([g[c] for c in GLOBAL_COLS], dtype=torch.float32),
        "native_iface": torch.full((n,), -1.0),             # filled in a second pass
        "res_keys": keys, "chain_seqs": {("H" if c == "H" else "L" if c == "L" else "AG"): "".join(s)
                                         for c, s in chain_seq.items()},
    }
    info = {"record_id": rid, "n_nodes": n, "n_edges": len(edges), "n_iface_res": len(iface_res),
            "csv_missing": csv_missing}
    if rec["label"] == 1:
        info["native_keys"] = {f"{c}|{k}": a for (c, k), a in zip(keys, aa.tolist())}
        # full-structure interface keys (not only cropped nodes)
        info["native_iface_keys"] = sorted(
            f"{residues[r][0] if residues[r][0] in ROLE else 'AG'}|{residues[r][1].id[1]}{residues[r][1].id[2].strip()}"
            for r in iface_res)
    return data, info


def process_record(rec, out_dir):
    data, info = build_graph(rec)
    torch.save(data, os.path.join(out_dir, "graphs", rec["record_id"].replace(":", "__") + ".pt"))
    return info


# ----------------------------------------------------------------------------- driver

def build_records(data_dir):
    recs = []
    wp = {}
    wp_path = os.path.join(data_dir, "docking_worst_poses.csv")
    if os.path.exists(wp_path):
        for r in csv.DictReader(open(wp_path)):
            wp[(r["set"], r["complex"])] = float(r["DockQ"])
    for short, folder, label in SETS:
        sdir = os.path.join(data_dir, folder, "Structures")
        for f in sorted(os.listdir(sdir)):
            cid = f[:-4]
            recs.append({"record_id": f"{short}:{cid}", "complex_id": cid, "pdb_code": cid.split("_")[0].lower(),
                         "set": short, "label": label,
                         "dockq": 1.0 if label else wp[(folder, cid)],
                         "pdb_path": os.path.join(sdir, f),
                         "csv_path": os.path.join(data_dir, folder, "Descriptors", cid + ".csv")})
    return recs


def make_splits(recs, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([r["label"] for r in recs]); g = np.array([r["pdb_code"] for r in recs])
    idx = np.arange(len(recs))
    trval, test = next(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(idx, y, g))
    split = {}
    for i in test:
        split[recs[i]["record_id"]] = "test"
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(trval, y[trval], g[trval])):
        for i in trval[va]:
            split[recs[i]["record_id"]] = str(k)
    return split


def compute_norm_stats(out_dir, split):
    """Per-fold statistics from TRAINING records only. split: record_id -> 'test' | '0'..'4'."""
    def stats(rids):
        xs, es, gs = [], [], []
        for rid in rids:
            d = torch.load(os.path.join(out_dir, "graphs", rid.replace(":", "__") + ".pt"), weights_only=False)
            xs.append(d["x"].numpy()); es.append(d["edge_attr"].numpy()); gs.append(d["global_feats"].numpy()[None])
        out = {}
        for name, arr, cols in (("node", np.concatenate(xs), NODE_COLS), ("edge", np.concatenate(es), EDGE_COLS),
                                ("global", np.concatenate(gs), GLOBAL_COLS)):
            out[name] = {c: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std() + 1e-6),
                             "p0.5": float(np.percentile(arr[:, i], 0.5)), "p99.5": float(np.percentile(arr[:, i], 99.5))}
                         for i, c in enumerate(cols)}
        out["n_records"] = len(rids)
        return out
    norm = {}
    for k in range(5):
        norm[f"fold{k}"] = stats([r for r, sp in split.items() if sp not in ("test", str(k))])
        print(f"norm stats fold {k}: {norm[f'fold{k}']['n_records']} training records", flush=True)
    norm["final"] = stats([r for r, sp in split.items() if sp != "test"])
    json.dump(norm, open(os.path.join(out_dir, "norm_stats.json"), "w"), indent=1)
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resplit", action="store_true", help="ignore an existing splits.csv and make new splits")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "graphs"), exist_ok=True)
    t0 = time.time()

    recs = build_records(a.data)
    if a.limit:
        recs = recs[:a.limit // 2] + [r for r in recs if r["label"] == 0][:a.limit // 2]
    split_path = os.path.join(a.out, "splits.csv")
    if os.path.exists(split_path) and not a.resplit:       # keep existing assignments stable across re-runs
        old = {r["record_id"]: r["split"] for r in csv.DictReader(open(split_path))}
        recs = [r for r in recs if r["record_id"] in old]
        split = {r["record_id"]: old[r["record_id"]] for r in recs}
        print(f"reusing splits.csv for {len(recs)} records", flush=True)
    else:
        split = make_splits(recs)

    # pass 1: graphs
    infos, failures = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_record, r, a.out): r["record_id"] for r in recs}
        for k, fu in enumerate(as_completed(futs), 1):
            rid = futs[fu]
            try:
                infos[rid] = fu.result()
            except Exception as e:
                failures[rid] = f"{type(e).__name__}: {e}"
            if k % 250 == 0:
                print(f"{time.strftime('%H:%M:%S')} {k}/{len(recs)} done, {len(failures)} failed", flush=True)

    ok_recs = [r for r in recs if r["record_id"] in infos]
    with open(os.path.join(a.out, "records.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]) + ["split", "graph"]); w.writeheader()
        for r in ok_recs:
            w.writerow({**r, "split": split[r["record_id"]],
                        "pdb_path": os.path.relpath(r["pdb_path"], a.out), "csv_path": os.path.relpath(r["csv_path"], a.out),
                        "graph": "graphs/" + r["record_id"].replace(":", "__") + ".pt"})
    with open(split_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["record_id", "pdb_code", "label", "split"])
        for r in ok_recs:
            w.writerow([r["record_id"], r["pdb_code"], r["label"], split[r["record_id"]]])
    print(f"{len(ok_recs)} records with graphs; split counts:",
          dict(Counter((split[r["record_id"]], r["label"]) for r in ok_recs)), flush=True)

    # pass 2: native-interface labels (positives: own interface; negatives: matched crystal, if residues agree)
    native = {r["complex_id"]: infos[r["record_id"]] for r in recs if r["label"] == 1 and r["record_id"] in infos}
    matched = Counter()
    for r in recs:
        rid = r["record_id"]
        if rid not in infos:
            continue
        p = os.path.join(a.out, "graphs", rid.replace(":", "__") + ".pt")
        d = torch.load(p, weights_only=False)
        if r["label"] == 1:
            d["native_iface"] = d["x"][:, NODE_COLS.index("interface")].clone()
            matched["positive"] += 1
        elif r["complex_id"] in native:
            nat = native[r["complex_id"]]
            keys = [f"{c}|{k}" for c, k in d["res_keys"]]
            common = [i for i, k in enumerate(keys) if k in nat["native_keys"]]
            agree = sum(nat["native_keys"][keys[i]] == int(d["aa"][i]) for i in common)
            if common and agree / len(common) >= 0.9 and len(common) / len(keys) >= 0.5:
                iface = set(nat["native_iface_keys"])
                d["native_iface"] = torch.tensor([1.0 if k in iface else 0.0 for k in keys])
                matched["negative_matched"] += 1
            else:
                matched["negative_numbering_mismatch"] += 1
        else:
            matched["negative_no_crystal"] += 1
        torch.save(d, p)

    # sequences
    with open(os.path.join(a.out, "sequences.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["record_id", "chain_role", "sequence"])
        for r in recs:
            rid = r["record_id"]
            if rid in infos:
                d = torch.load(os.path.join(a.out, "graphs", rid.replace(":", "__") + ".pt"), weights_only=False)
                for role_name in ("H", "L", "AG"):
                    w.writerow([rid, role_name, d["chain_seqs"][role_name]])

    compute_norm_stats(a.out, {r["record_id"]: split[r["record_id"]] for r in recs if r["record_id"] in infos})

    json.dump({"node_cols": NODE_COLS, "edge_cols": EDGE_COLS, "edge_types": EDGE_TYPES, "global_cols": GLOBAL_COLS,
               "aa": AA3 + ["UNK"], "ss": list(SS_CODES) + ["missing"], "roles": ["H", "L", "AG"], "regions": REGIONS,
               "coords": ["N", "CA", "C", "O", "CB_virtual"],
               "edge_index": "row 0 = sender j, row 1 = receiver i; geometry is expressed in receiver frame",
               "params": {"iface_cut": IFACE_CUT, "hb_cut": HB_CUT, "sb_cut": SB_CUT, "crop_cut": CROP_CUT,
                          "neigh_cut": NEIGH_CUT, "max_nodes": MAX_NODES, "knn": KNN, "knn_radius": KNN_RADIUS,
                          "seq_k": SEQ_K, "cdr_definition": "Chothia CDRs on Martin numbering", "split_seed": 0}},
              open(os.path.join(a.out, "schema.json"), "w"), indent=1)
    summary = {"records": len(recs), "graphs": len(infos), "failures": failures, "native_labels": dict(matched),
               "nodes_mean": float(np.mean([i["n_nodes"] for i in infos.values()])),
               "nodes_max": int(max(i["n_nodes"] for i in infos.values())),
               "csv_missing_residues": int(sum(i["csv_missing"] for i in infos.values())),
               "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "preprocess_log.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), "failures:", len(failures), flush=True)


if __name__ == "__main__":
    main()
