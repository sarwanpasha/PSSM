# GPU queue 5: waits for gpu_queue4, then recomputes GB1 wild-type masked marginals with the correct WT background.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\gpu_queue5.log"
$q4 = Get-Content "$root\logs\gpu_queue4.pid"
while (Get-Process -Id $q4 -ErrorAction SilentlyContinue) { Start-Sleep 60 }
& $py -u "$code\extract_lm_logits.py" esm2_t33_650M_UR50D gb1_sampled $root wt 2>&1 | Out-File -Append -Encoding utf8 $log
Copy-Item "$root\emb\esm2_t33_650M_UR50D__gb1_sampled__wtlogp.npy" "$code\..\stage\emb\esm2_t33_650M_UR50D__gb1_wtlogp_fixed.npy" -Force
"GPU QUEUE5 DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
