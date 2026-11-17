# Keeps the system awake (not the display) while any experiment queue is running; exits when they finish.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$root = "$env:USERPROFILE\pssm_work\logs"
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
while ($true) {
  $alive = @("esm_queue.pid", "kernel_queue.pid", "variants_queue.pid") | ForEach-Object {
    $p = Get-Content "$root\$_" -ErrorAction SilentlyContinue; if ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue)) { $p } }
  if (-not $alive) { break }
  [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
  Start-Sleep 120
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
"keep-awake exited $(Get-Date)" | Out-File -Append "$root\keep_awake.log"
