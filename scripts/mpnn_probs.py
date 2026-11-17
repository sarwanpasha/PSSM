"""Structure-conditioned per-position amino acid distributions from ProteinMPNN.
Loads the ProteinMPNN model by explicit path and runs one forward pass to obtain
p(residue | backbone) at every position, the same signal Kermut derives from an
inverse-folding model.
usage: python mpnn_probs.py <mpnn_dir> <pdb_path> <chain> <out_npy>
Writes an (L,20) array of log probabilities in ACDEFGHIKLMNPQRSTVWY order.
"""
import sys, os
import numpy as np
import torch

mpnn_dir, pdb_path, chain, out_npy = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, mpnn_dir)
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize, StructureDatasetPDB

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"
dev = torch.device("cpu")

ckpt_path = os.path.join(mpnn_dir, "vanilla_model_weights", "v_48_020.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hidden = 128
model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden, edge_features=hidden,
                    hidden_dim=hidden, num_encoder_layers=3, num_decoder_layers=3,
                    augment_eps=0.0, k_neighbors=ckpt["num_edges"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(dev).eval()
print("loaded ProteinMPNN, k_neighbors", ckpt["num_edges"], flush=True)

pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
batch = [dataset[0]]
chain_id_dict = {pdb_dict_list[0]["name"]: ([chain], [])}
out = tied_featurize(batch, dev, chain_id_dict, None, None, None, None, None, ca_only=False)
X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx = out[0], out[1], out[2], out[3], out[4], out[5], out[12]

with torch.no_grad():
    log_p = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
log_p = log_p[0].cpu().numpy()                      # (L, 21)
idx = [MPNN_ALPHABET.index(a) for a in PROTEIN]
lp = log_p[:, idx]
lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))  # renormalise over the 20 residues
np.save(out_npy, lp.astype(np.float32))
native = "".join(MPNN_ALPHABET[i] for i in S[0].cpu().numpy())
print("wrote", out_npy, lp.shape, flush=True)
print("native", native, flush=True)
print("argmax", "".join(PROTEIN[i] for i in lp.argmax(1)), flush=True)
