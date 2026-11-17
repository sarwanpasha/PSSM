# GPU queue 4: per-residue ESM-2 650M log-probs for LM-weighted W-PSSKM.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\gpu_queue4.log"
foreach ($d in @("gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high")) {
  & $py -u "$code\extract_lm_logits.py" esm2_t33_650M_UR50D $d $root wt 2>&1 | Out-File -Append -Encoding utf8 $log }
foreach ($d in @("deeploc", "tape_rh_fold")) {
  & $py -u "$code\extract_lm_logits.py" esm2_t33_650M_UR50D $d $root 2>&1 | Out-File -Append -Encoding utf8 $log }
"GPU QUEUE4 DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
