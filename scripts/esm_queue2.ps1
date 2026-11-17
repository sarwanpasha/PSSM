# Restart of the GPU queue for the remaining ESM-2 650M datasets (GB1 already done).
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
foreach ($d in @("tape_rh_fold", "tape_rh_superfamily", "tape_rh_family", "deeploc", "deeploc_random")) {
  if (Test-Path "$root\emb\esm2_t33_650M_UR50D__$d.npy") { continue }
  & $py -u "$code\extract_esm.py" esm2_t33_650M_UR50D $d $root 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\esm_queue.log"
}
"QUEUE2 DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\esm_queue.log"
