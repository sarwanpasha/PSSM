"""Print mean metric per representation for strings_*.json and combo_*.json (partial results OK)."""
import json, glob, os, sys, collections
r = os.path.join(sys.argv[1], "results")
for pat in ["strings_*.json", "combo_*.json"]:
    for f in sorted(glob.glob(os.path.join(r, pat))):
        d = json.load(open(f)); agg = collections.defaultdict(list)
        for x in d["runs"]:
            m = x.get("acc", x.get("spearman")); t = x.get("test", "test")
            agg[x["kernel"] + ("" if t == "test" else "/" + t)].append(m)
        print(os.path.basename(f)[:-5], " | ".join(f"{k}={sum(v)/len(v):.3f}({len(v)})" for k, v in agg.items()))
