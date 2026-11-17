# GPU queue 7: partial fine-tune of ESM-2 (last 2 blocks + head) on the GB1 splits.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\finetune.log"
foreach ($s in @("two_vs_rest", "one_vs_rest", "three_vs_rest", "low_vs_high", "sampled")) {
  "=== esm2_t12_35M_UR50D $s $(Get-Date)" | Out-File -Append -Encoding utf8 $log
  & $py -u "$code\finetune_esm.py" esm2_t12_35M_UR50D $s $root 8 3e-4 8 2 2>&1 | Out-File -Append -Encoding utf8 $log }
foreach ($s in @("two_vs_rest", "three_vs_rest")) {
  "=== esm2_t30_150M_UR50D $s $(Get-Date)" | Out-File -Append -Encoding utf8 $log
  & $py -u "$code\finetune_esm.py" esm2_t30_150M_UR50D $s $root 8 3e-4 8 2 2>&1 | Out-File -Append -Encoding utf8 $log }
"FINETUNE DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
