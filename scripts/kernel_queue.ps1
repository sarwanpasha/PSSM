# Background CPU queue for the kernel experiments.
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
$jobs = @(
  @("gb1_sampled", ""), @("gb1_one_vs_rest", ""), @("gb1_two_vs_rest", ""), @("gb1_three_vs_rest", ""), @("gb1_low_vs_high", ""),
  @("gb_nontata", ""), @("gb_cohn", ""),
  @("tape_rh_fold", "--no-original"), @("tape_rh_superfamily", "--no-original"), @("tape_rh_family", "--no-original"),
  @("deeploc", ""), @("deeploc_random", ""),
  @("meltome_mixed_split", "--no-original")
)
foreach ($j in $jobs) {
  $d = $j[0]
  if (Test-Path "$root\results\kernels_$d.json") { $r = Get-Content "$root\results\kernels_$d.json" -Raw; if ($r -match '"fold": 4' -or ($d -notlike "deeploc*")) { if (Select-String -Path "$root\logs\kernels_$d.log" -Pattern "done" -Quiet -ErrorAction SilentlyContinue) { continue } } }
  $a = @("-u", "$code\run_kernels.py", $d, $root); if ($j[1]) { $a += $j[1] }
  & $py @a 2>&1 | Out-File -Encoding utf8 "$root\logs\kernels_$d.log"
}
"KERNEL QUEUE DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\kernel_queue.log"
