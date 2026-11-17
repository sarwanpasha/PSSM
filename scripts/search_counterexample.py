import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm import encode, gram_original
rng = np.random.default_rng(7)

def score(seqs, alpha):
    K = gram_original(encode(seqs, alpha), len(alpha))
    ev = np.linalg.eigvalsh(K)
    return ev[0] / max(ev[-1], 1e-12)

def mutate(seqs, alpha, maxlen):
    s = list(seqs); i = rng.integers(len(s)); x = list(s[i]); r = rng.random()
    if r < 0.4 and x: x[rng.integers(len(x))] = rng.choice(list(alpha))
    elif r < 0.7 and len(x) < maxlen: x.insert(rng.integers(len(x)+1), rng.choice(list(alpha)))
    elif len(x) > 1: x.pop(rng.integers(len(x)))
    s[i] = "".join(x); return s

overall = 0.0; best_set = None
for restart in range(400):
    alpha = "ACDEFGHIKLMNPQRSTVWY"[: int(rng.integers(2, 6))]
    n = int(rng.integers(2, 8)); maxlen = int(rng.integers(2, 15))
    cur = ["".join(rng.choice(list(alpha), int(rng.integers(1, maxlen+1)))) for _ in range(n)]
    cs = score(cur, alpha)
    for it in range(400):
        cand = mutate(cur, alpha, maxlen); sc = score(cand, alpha)
        if sc <= cs: cur, cs = cand, sc
    if cs < overall: overall, best_set = cs, (alpha, cur)
print("most negative lambda_min/lambda_max found:", overall)
print("set:", best_set)
