# Background GPU queue: small models first so every dataset gets a baseline early.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
$datasets = @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high", "tape_rh_fold", "tape_rh_superfamily", "tape_rh_family", "deeploc", "deeploc_random")
foreach ($m in @("esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D")) {
  foreach ($d in $datasets) {
    if ($d -like "gb_*") { continue }   # ESM is a protein model; skip DNA
    & $py -u "$code\extract_esm.py" $m $d $root 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\esm_queue.log"
  }
}
"QUEUE DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\esm_queue.log"
