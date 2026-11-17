# Runs after the kernel queue finishes (CPU-bound jobs one at a time).
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
$kq = Get-Content "$root\logs\kernel_queue.pid" -ErrorAction SilentlyContinue
while ($kq -and (Get-Process -Id $kq -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
foreach ($d in @("gb1_sampled", "gb1_one_vs_rest", "gb1_two_vs_rest", "gb1_three_vs_rest", "gb1_low_vs_high",
                 "gb_nontata", "gb_cohn", "tape_rh_fold", "tape_rh_superfamily", "tape_rh_family", "deeploc", "deeploc_random", "meltome_mixed_split")) {
  if (Test-Path "$root\results\variants_$d.json") { continue }
  & $py -u "$code\run_variants.py" $d $root 2>&1 | Out-File -Encoding utf8 "$root\logs\variants_$d.log"
}
& $py -u "$code\run_esm_baselines.py" $root 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\esm_baselines.log"
& $py -u "$code\summarize.py" $root 2>&1 | Out-File -Encoding utf8 "$root\logs\summary.log"
"VARIANTS QUEUE DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\variants_queue.log"
