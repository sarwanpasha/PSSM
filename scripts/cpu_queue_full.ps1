# Full-data Gaussian product kernel (Nystrom) + AAV, on CPU.
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"; $env:OPENBLAS_NUM_THREADS = "4"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
foreach ($d in @("gb_nontata", "gb_cohn", "deeploc", "tape_rh_family", "tape_rh_superfamily", "tape_rh_fold", "meltome_mixed_split")) {
  & $py -u "$code\run_gauss_full.py" $d $root 2000 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\gaussfull.log" }
"GAUSSFULL DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\gaussfull.log"
