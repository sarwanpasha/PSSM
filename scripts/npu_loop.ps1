# Keep the NPU worker going on 650M DeepLoc (<=512 residues); restart after transient device errors (max 6 tries).
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = "$env:USERPROFILE\pssm_work"
for ($k = 1; $k -le 6; $k++) {
  if (Test-Path "$root\emb\esm2_t33_650M_UR50D__deeploc.npy") { break }
  "attempt $k $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\npu_650M_deeploc.log"
  & $py -u "$code\ov_extract.py" esm2_t33_650M_UR50D deeploc $root NPU 514 520 2>>"$root\logs\npu_650M_deeploc.err" | Out-File -Append -Encoding utf8 "$root\logs\npu_650M_deeploc.log"
  if ($LASTEXITCODE -eq 0) { break }
  Start-Sleep 30
}
"NPU LOOP DONE $(Get-Date)" | Out-File -Append -Encoding utf8 "$root\logs\npu_650M_deeploc.log"
