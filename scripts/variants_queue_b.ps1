# Second variants worker (parallel to variants_queue.ps1): takes DeepLoc and Meltome.
# Placeholder JSONs (empty runs) make the first worker skip these datasets.
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
$mine = @("meltome_mixed_split", "deeploc", "deeploc_random")
foreach ($d in $mine) { if (-not (Test-Path "$root\results\variants_$d.json")) { "{`"dataset`": `"$d`", `"runs`": [], `"placeholder`": true}" | Out-File -Encoding ascii "$root\results\variants_$d.json" } }
foreach ($d in $mine) {
  $j = Get-Content "$root\results\variants_$d.json" -Raw
  if ($j -notmatch '"placeholder"') { continue }
  & $py -u "$code\run_variants.py" $d $root 2>&1 | Out-File -Encoding utf8 "$root\logs\variants_$d.log"
}
"VARIANTS B DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\variants_queue.log"
