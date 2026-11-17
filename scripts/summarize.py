"""Collect results/*.json into results/summary.csv and a markdown table (mean ± sd over folds)."""
import sys, os, glob, json
import pandas as pd
root = sys.argv[1]; rows = []
for f in glob.glob(os.path.join(root, "results", "*.json")):
    r = json.load(open(f))
    for x in r["runs"]:
        rows.append(dict(dataset=r["dataset"], **x))
df = pd.DataFrame(rows)
metric_cols = [c for c in ["acc", "f1_weighted", "f1_macro", "spearman", "r2"] if c in df]
g = df.groupby(["dataset", "kernel", "model"])[metric_cols].agg(["mean", "std", "count"])
g.to_csv(os.path.join(root, "results", "summary.csv"))
lines = []
for ds, d in df.groupby("dataset"):
    m = "acc" if d["acc"].notna().any() else "spearman"
    s = d.groupby(["kernel", "model"])[m].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    lines.append(f"\n### {ds} ({m})\n\n| representation | model | mean | sd | folds |\n|---|---|---|---|---|")
    for (k, mo), v in s.iterrows():
        lines.append(f"| {k} | {mo} | {v['mean']:.3f} | {0 if pd.isna(v['std']) else v['std']:.3f} | {int(v['count'])} |")
for f in glob.glob(os.path.join(root, "results", "kernels_*.json")):
    r = json.load(open(f)); c = r.get("original_psd_check")
    if c: lines.append(f"\nPSD check {r['dataset']}: lambda_min={c['lambda_min']:.4g}, lambda_max={c['lambda_max']:.4g}, ratio={c['ratio']:.2e}")
open(os.path.join(root, "results", "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
