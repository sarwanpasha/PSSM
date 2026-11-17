# FLIP AAV splits.
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"; $env:OPENBLAS_NUM_THREADS = "4"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
foreach ($s in @("two_vs_many", "seven_vs_many", "one_vs_many", "low_vs_high", "sampled", "mut_des", "des_mut")) {
  & $py -u "$code\run_aav.py" $s $root 4000 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\aav.log" }
"AAV DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\aav.log"
