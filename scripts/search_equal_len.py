import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm import encode, gram_original
rng = np.random.default_rng(11)
def score(s, a):
    ev = np.linalg.eigvalsh(gram_original(encode(s, a), len(a))); return ev[0]/ev[-1]
overall, best = 0.0, None
for restart in range(400):
    a = "ACDEFGHIKLMNPQRSTVWY"[: int(rng.integers(2, 6))]
    n = int(rng.integers(2, 8)); L = int(rng.integers(2, 12))
    cur = ["".join(rng.choice(list(a), L)) for _ in range(n)]; cs = score(cur, a)
    for it in range(400):
        c = list(cur); i = rng.integers(n); x = list(c[i]); x[rng.integers(L)] = rng.choice(list(a)); c[i] = "".join(x)
        sc = score(c, a)
        if sc <= cs: cur, cs = c, sc
    if cs < overall: overall, best = cs, (a, cur)
print("equal-length: most negative lambda_min/lambda_max:", overall, best)
