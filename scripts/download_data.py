"""Download benchmark datasets into <root>/data. usage: python download_data.py <root>"""
import os, sys, urllib.request, zipfile, tarfile, shutil
root = sys.argv[1]; D = os.path.join(root, "data"); os.makedirs(D, exist_ok=True)

def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("have", path); return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("get ", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(path + ".part", path); print("  ->", os.path.getsize(path), "bytes", flush=True)
    return path

jobs = {
 "deeploc": [("https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv", "Swissprot_Train_Validation_dataset.csv"),
             ("https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/hpa_testset.csv", "hpa_testset.csv")],
 "flip_gb1": [("https://github.com/J-SNACKKB/FLIP/raw/main/splits/gb1/splits.zip", "splits.zip")],
 "flip_meltome": [("https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/splits.zip", "splits.zip")],
 "tape_remote_homology": [("http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/remote_homology.tar.gz", "remote_homology.tar.gz")],
}
gb = "https://huggingface.co/datasets/katielink/genomic-benchmarks/resolve/main"
for ds, classes in [("human_nontata_promoters", ["negative", "positive"]),
                    ("human_enhancers_cohn", ["negative", "positive"]),
                    ("human_ensembl_regulatory", ["enhancer", "ocr", "promoter"])]:
    jobs["gb_" + ds] = [(f"{gb}/{ds}/{sp}/{c}.csv.gz", f"{sp}/{c}.csv.gz") for sp in ["train", "test"] for c in classes]

only = set(sys.argv[2:])
for ds, files in jobs.items():
    if only and ds not in only: continue
    for url, rel in files:
        try:
            p = get(url, os.path.join(D, ds, rel))
            if p.endswith(".zip"):
                with zipfile.ZipFile(p) as z: z.extractall(os.path.dirname(p))
            if p.endswith(".tar.gz"):
                with tarfile.open(p) as t: t.extractall(os.path.dirname(p), filter="data")
        except Exception as e:
            print("FAILED", ds, url, type(e).__name__, e, flush=True)
print("done")
