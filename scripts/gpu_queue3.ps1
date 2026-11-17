# GPU queue: binned weighted ESM-2 35M features, then Nucleotide Transformer embeddings for DNA.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\gpu_queue3.log"
foreach ($d in @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high", "tape_rh_fold", "tape_rh_superfamily", "tape_rh_family", "deeploc")) {
  if (Test-Path "$root\emb\esm2_t12_35M_UR50D__${d}__bin.npy") { continue }
  & $py -u "$code\extract_esm_binned.py" esm2_t12_35M_UR50D $d $root 2>&1 | Out-File -Append -Encoding utf8 $log
}
foreach ($d in @("gb_nontata", "gb_cohn")) {
  if (Test-Path "$root\emb\nt500m__$d.npy") { continue }
  & $py -u "$code\extract_dna_lm.py" $d $root 2>&1 | Out-File -Append -Encoding utf8 $log
}
"GPU QUEUE3 DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
