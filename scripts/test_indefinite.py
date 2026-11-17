"""Search for indefiniteness of the original kernel in realistic regimes."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm import *
rng = np.random.default_rng(1)
A = len(PROTEIN)

def family(n, L, mut, alpha=PROTEIN):
    base = rng.choice(list(alpha), L)
    out = []
    for _ in range(n):
        s = base.copy()
        k = max(1, int(mut * L))
        pos = rng.choice(L, k, replace=False)
        s[pos] = rng.choice(list(alpha), k)
        out.append("".join(s))
    return out

def trial(gen, trials=300):
    neg, worst = 0, 0.0
    for _ in range(trials):
        s = gen()
        X = encode(s)
        ev = np.linalg.eigvalsh(gram_original(X, A))
        rel = ev[0] / ev[-1]
        worst = min(worst, rel); neg += ev[0] < -1e-9
    return neg, trials, worst

regimes = {
    "random, L=4-12": lambda: ["".join(rng.choice(list(PROTEIN), int(rng.integers(4, 12)))) for _ in range(5)],
    "family L=50, 10% mutated": lambda: family(8, 50, 0.10),
    "family L=50, 30% mutated": lambda: family(8, 50, 0.30),
    "family L=20, 30% mutated": lambda: family(8, 20, 0.30),
    "unequal lengths (prefixes of one seq)": lambda: (lambda b: [b[:int(rng.integers(5, len(b)+1))] for _ in range(6)])("".join(rng.choice(list(PROTEIN), 40))),
    "low-complexity (3-letter) L=10": lambda: ["".join(rng.choice(list("AGS"), 10)) for _ in range(6)],
}
for name, g in regimes.items():
    neg, t, worst = trial(g)
    print(f"{name:42s}: {neg:3d}/{t} indefinite, worst lambda_min/lambda_max = {worst:.3f}")

# explicit tiny counterexample search
best = None
for _ in range(20000):
    L = int(rng.integers(2, 6))
    s = ["".join(rng.choice(list("ACD"), int(rng.integers(1, L+1)))) for _ in range(3)]
    ev = np.linalg.eigvalsh(gram_original(encode(s), A))
    if best is None or ev[0] < best[0]:
        best = (ev[0], s)
print("smallest counterexample found:", best[1], "lambda_min =", round(best[0], 4))
print(np.round(gram_original(encode(best[1]), A), 4))
