# Product kernels on weighted position-specific features

Code for the paper *When Sequence Kernels Should Multiply, Not Add*, the journal
extension of "Position Specific Scoring Is All You Need? Revisiting Protein Sequence
Classification Tasks" (RECOMB-CG 2025, LNCS 15666, pp. 202-215).

The repository implements a Gaussian product kernel on weighted position-specific
features, together with the attribution tools and the experiments reported in the paper.

## Method in one paragraph

A sequence `x` of length `L` over an alphabet of size `A` is mapped to a sparse
`L x A` matrix whose entry `(t,a)` is `sqrt(w(t,a))` when `x_t = a` and zero otherwise.
The linear kernel on this map is a weighted count of matching positions, and the
Gaussian kernel `k_g(x,y) = exp(-g * ||phi(x) - phi(y)||^2)` turns the sum over
positions into a product, so the model can represent interactions between sites.
Three weightings are supported: uniform, within-sequence composition, and the
masked-marginal probabilities of a protein language model.

## Layout

```
scripts/       data download, language-model feature extraction, benchmark runners,
               and the PowerShell queues used to drive long jobs
experiments/   the experiments reported in the paper, one file per table or figure
```

Key entry points:

| File | Purpose |
|---|---|
| `scripts/download_data.py` | fetch FLIP, TAPE, DeepLoc and Genomic Benchmarks into `<root>/data` |
| `scripts/extract_wt_marginals.py` | ESM-2 wild-type masked marginals used as kernel weights |
| `scripts/extract_lm_logits.py` | per-residue masked log-probabilities |
| `scripts/run_kernels.py` | kernel baselines on the classification benchmarks |
| `scripts/run_gauss_full.py` | full-data product kernel with the Nystrom approximation |
| `scripts/run_aav.py` | FLIP AAV capsid splits |
| `experiments/gb1.py` | GB1 splits, product kernel against baselines |
| `experiments/ablate.py` | weighting ablation under a single selection protocol |
| `experiments/zhou.py` | per-site decay fitted by Gaussian process marginal likelihood |
| `experiments/pg.py`, `experiments/pg2.py` | ProteinGym extrapolation splits |
| `experiments/posimp.py` | position knock-out attribution |
| `experiments/theory_check.py` | numerical verification of the rank ceiling and strict positive definiteness |
| `experiments/make_figures.py` | figures |

## Requirements

Python 3.10 or later with `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`,
`torch`, `fair-esm`, and `transformers`. See `requirements.txt`.

## Usage

```bash
python scripts/download_data.py <root>
python scripts/extract_wt_marginals.py <root>
python experiments/gb1.py <root>
```

`<root>` is a working directory of your choice; the scripts create `data/`, `emb/`
and `results/` beneath it. The benchmark datasets are not redistributed here and are
downloaded from their original sources.

## Data

| Benchmark | Source |
|---|---|
| FLIP (GB1, AAV, Meltome) | https://github.com/J-SNACKKB/FLIP |
| ProteinGym | https://proteingym.org |
| TAPE remote homology | https://github.com/songlab-cal/tape |
| DeepLoc 2.0 | https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ |
| Genomic Benchmarks | https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks |

## Citation

```bibtex
@inproceedings{ali2025pssm,
  author    = {Ali, Sarwan and Murad, Taslim and Chourasia, Prakash and Mansoor, Haris
               and Khan, Imdad Ullah and Chen, Pin-Yu and Patterson, Murray},
  title     = {Position Specific Scoring Is All You Need? Revisiting Protein Sequence
               Classification Tasks},
  booktitle = {Comparative Genomics (RECOMB-CG 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15666},
  pages     = {202--215},
  publisher = {Springer},
  year      = {2025}
}
```
