"""Download a PDB entry. usage: python get_pdb.py <id> <outdir>"""
import sys, os, urllib.request
pid, outdir = sys.argv[1].upper(), sys.argv[2]
os.makedirs(outdir, exist_ok=True)
url = "https://files.rcsb.org/download/%s.pdb" % pid
dst = os.path.join(outdir, pid + ".pdb")
urllib.request.urlretrieve(url, dst)
n = sum(1 for l in open(dst) if l.startswith("ATOM"))
seq = []
seen = set()
three = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
         "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
         "TYR":"Y","VAL":"V"}
for l in open(dst):
    if l.startswith("ATOM") and l[12:16].strip() == "CA":
        key = (l[21], l[22:27])
        if key in seen:
            continue
        seen.add(key)
        seq.append(three.get(l[17:20].strip(), "X"))
print("wrote", dst, "atoms", n, "residues", len(seq))
print("seq", "".join(seq))
