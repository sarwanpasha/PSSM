"""Stream a TAPE remote-homology json into a slim csv (primary, fold_label) without loading it in memory."""
import re, sys, csv
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(r'"(primary|fold_label)":\s*(?:"([A-Za-z]+)"|(\d+))')
prim, lab, tail = [], [], ""
with open(src, "r") as f:
    while True:
        chunk = f.read(1 << 24)
        if not chunk: break
        buf = tail + chunk; last = 0
        for m in pat.finditer(buf):
            if m.end() > len(buf) - 200000 and len(chunk) == 1 << 24: break   # may be cut: leave for next round
            (prim if m.group(1) == "primary" else lab).append(m.group(2) or m.group(3)); last = m.end()
        tail = buf[last:]
assert len(prim) == len(lab), (len(prim), len(lab))
with open(dst, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["primary", "fold_label"]); w.writerows(zip(prim, lab))
print(len(prim), "records ->", dst)
