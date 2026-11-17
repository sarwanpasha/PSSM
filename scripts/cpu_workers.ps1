# One CPU worker; usage: cpu_workers.ps1 <name> ; each worker limited to 2 BLAS threads.
param([string]$which)
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\worker_$which.log"
switch ($which) {
  "strings1" { foreach ($d in @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high", "gb_nontata", "gb_cohn", "deeploc")) {
                 & $py -u "$code\run_stringkernels.py" $d $root 2>&1 | Out-File -Append -Encoding utf8 $log } }
  "strings2" { & $py -u "$code\run_stringkernels.py" tape_rh $root spec3 spec4 mm3 wd2 2>&1 | Out-File -Append -Encoding utf8 $log }
  "combo"    { foreach ($d in @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high", "deeploc", "tape_rh")) {
                 & $py -u "$code\run_combo.py" $d $root esm2_t33_650M_UR50D 2>&1 | Out-File -Append -Encoding utf8 $log } }
  "combobin" { foreach ($d in @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high", "deeploc", "tape_rh")) {
                 & $py -u "$code\run_combo.py" $d $root esm2_t12_35M_UR50D esm2_t12_35M_UR50D 2>&1 | Out-File -Append -Encoding utf8 $log } }
}
"WORKER $which DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
