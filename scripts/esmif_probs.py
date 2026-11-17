"""Structure-conditioned per-position residue distributions from ESM-IF1.

ESM-IF1 is an inverse-folding model: it predicts residues from a backbone. It is autoregressive,
so the distribution at position t is taken with the backbone given and the native prefix supplied,
which is how ESM-IF1 is used for variant scoring. The result is the structure-conditioned analogue
of the masked marginals we read from a sequence model.

Two compatibility notes. ESM-IF1 imports two names from biotite that were renamed in biotite 2.x,
so they are aliased before the import. The backbone coordinates are parsed here rather than through
esm's own loader, so the only part of esm.inverse_folding this depends on is the model itself.

usage: python esmif_probs.py <pdb_path> <chain> <out_npy> [shim_dir]
Writes an (L,20) array of log probabilities in ACDEFGHIKLMNPQRSTVWY order.
"""
import sys, os, time
import numpy as np
import torch

shim_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shims")
sys.path.insert(0, shim_dir)

pdb_path, chain, out_npy = sys.argv[1], sys.argv[2], sys.argv[3]
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

import biotite.structure as _bs
if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone
if not hasattr(_bs, "get_chains"):
    _bs.get_chains = lambda a: np.unique(a.chain_id)

import esm
import esm.inverse_folding as inv


def parse_backbone(path, chain_id):
    """N, CA, C coordinates and the one-letter sequence, in residue order."""
    res = {}
    order = []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain_id:
            continue
        atom = line[12:16].strip()
        if atom not in ("N", "CA", "C"):
            continue
        key = line[22:27]
        if key not in res:
            res[key] = {"name": line[17:20].strip()}
            order.append(key)
        res[key][atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    coords, seq = [], []
    for key in order:
        r = res[key]
        if not all(a in r for a in ("N", "CA", "C")):
            continue
        coords.append([r["N"], r["CA"], r["C"]])
        seq.append(THREE.get(r["name"], "X"))
    return np.array(coords, dtype=np.float32), "".join(seq)


coords, native_seq = parse_backbone(pdb_path, chain)
print("residues", len(native_seq), "coords", coords.shape, flush=True)
print("native  ", native_seq, flush=True)

t0 = time.time()
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()
print("loaded ESM-IF1 in %.0fs" % (time.time() - t0), flush=True)

batch_converter = inv.util.CoordBatchConverter(alphabet)
coords_t, confidence, strs, tokens, padding_mask = batch_converter(
    [(coords, None, native_seq)], device=torch.device("cpu"))
prev_output_tokens = tokens[:, :-1]
with torch.no_grad():
    logits, _ = model.forward(coords_t, padding_mask, confidence, prev_output_tokens)
lp_full = torch.log_softmax(logits[0].float(), dim=0).numpy()      # (vocab, L)
idx = [alphabet.get_idx(a) for a in PROTEIN]
lp = lp_full[idx, :].T[: len(native_seq)]                          # (L, 20)
lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))                 # renormalise over the 20 residues
np.save(out_npy, lp.astype(np.float32))
pred = "".join(PROTEIN[i] for i in lp.argmax(1))
rec = sum(a == b for a, b in zip(pred, native_seq)) / len(native_seq)
print("wrote", out_npy, lp.shape, flush=True)
print("argmax  ", pred, flush=True)
print("native sequence recovery %.3f" % rec, flush=True)
