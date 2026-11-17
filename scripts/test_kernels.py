"""Checks: vectorised == literal Algorithm 1; original kernel not PSD; corrected PSD."""
import numpy as np, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from wpsskm import *

rng = np.random.default_rng(0)
A = len(PROTEIN)

def rand_seqs(n, L):
    return ["".join(rng.choice(list(PROTEIN), L)) for _ in range(n)]

# 1. vectorised original equals the literal algorithm
for n, L in [(5, 6), (30, 40), (50, 200)]:
    s = rand_seqs(n, L)
    K1 = gram_original_naive(s); K2 = gram_original(encode(s), A)
    assert np.allclose(K1, K2), "mismatch"
print("vectorised original kernel == literal Algorithm 1: OK")

# 2. how often is the original Gram matrix indefinite?
worst, neg = 0.0, 0
trials = 500
for _ in range(trials):
    s = rand_seqs(5, int(rng.integers(4, 12)))
    ev = np.linalg.eigvalsh(gram_original(encode(s), A))
    worst = min(worst, ev[0]); neg += ev[0] < -1e-9
print(f"original kernel: {neg}/{trials} random 5-seq Gram matrices have a negative eigenvalue (min {worst:.3f})")

# 3. corrected kernels are PSD
for kind in ["product", "sqrt", "match"]:
    m = 0.0
    for _ in range(trials):
        s = rand_seqs(5, int(rng.integers(4, 12)))
        m = min(m, np.linalg.eigvalsh(gram_corrected(encode(s), A, kind))[0])
    print(f"corrected '{kind}': min eigenvalue over {trials} trials = {m:.2e}")
    assert m > -1e-9

# 4. full-rank claim: duplicate sequences give identical rows
s = rand_seqs(4, 10); s.append(s[0])
print("rank with a duplicate sequence:", np.linalg.matrix_rank(gram_original(encode(s), A)), "of", len(s))
