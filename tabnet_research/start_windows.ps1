param(
    [string]$SourceUrl = 'http://192.168.1.96:8765',
    [string]$WorkDirectory = 'C:\AvitoTabNet\EXP-TABNET-001-r01',
    [string]$BasePython,
    [ValidateSet('cu126','cu128')][string]$CudaWheel = 'cu126'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskName = 'AvitoTabNet001'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principalCheck = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principalCheck.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an elevated PowerShell window.'
}
New-Item -ItemType Directory -Force $WorkDirectory | Out-Null
$drive = (Get-Item -LiteralPath $WorkDirectory).PSDrive
if ($drive.Free -lt 25GB) { throw 'At least 25 GiB free disk space required in the selected workspace.' }
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old) { throw "Task $taskName already exists: inspect it before repeating setup." }
$sourceMeta = Invoke-RestMethod "$SourceUrl/source-transfer.json"
$sourceZip = Join-Path $WorkDirectory 'tabnet-source.zip'
Invoke-WebRequest -UseBasicParsing "$SourceUrl/tabnet-source.zip" -OutFile $sourceZip
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $sourceZip).Hash.ToLowerInvariant() -ne $sourceMeta.sha256) {
    throw 'Source package checksum mismatch'
}
Expand-Archive -LiteralPath $sourceZip -DestinationPath $WorkDirectory -Force
$bootstrap = Join-Path $WorkDirectory 'tabnet_research\bootstrap_windows.ps1'
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$bootstrap`" -CudaWheel $CudaWheel"
if ($BasePython) { $arguments += " -BasePython `"$BasePython`"" }
$entry = Join-Path $WorkDirectory 'scheduled_entry.ps1'
$escapedRoot = $WorkDirectory.Replace("'", "''")
$escapedArguments = $arguments.Replace("'", "''")
@"
`$ErrorActionPreference = 'Stop'
`$worker = Start-Process -FilePath 'powershell.exe' -ArgumentList '$escapedArguments' -WorkingDirectory '$escapedRoot' -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput '$escapedRoot\worker.log' -RedirectStandardError '$escapedRoot\worker.err.log'
exit `$worker.ExitCode
"@ | Set-Content -Encoding UTF8 -LiteralPath $entry
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$entry`"" -WorkingDirectory $WorkDirectory
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "Detached task $taskName started. Dependencies install while the cache transfers."
$cacheMeta = $null
$deadline = (Get-Date).AddMinutes(15)
while (-not $cacheMeta) {
    try { $cacheMeta = Invoke-RestMethod "$SourceUrl/cache-transfer.json" } catch {
        if ((Get-Date) -gt $deadline) { throw 'Mac cache package not ready within 15 minutes' }
        Write-Output 'Waiting for Mac to finish the cache package...'
        Start-Sleep -Seconds 5
    }
}
$cacheZip = Join-Path $WorkDirectory 'research-cache-train4-v1.zip'
Write-Output ("Copying {0:N2} GiB cache from Mac. Keep the laptop connected until verification finishes." -f ($cacheMeta.bytes / 1GB))
& curl.exe --fail --location --retry 2 --output $cacheZip "$SourceUrl/research-cache-train4-v1.zip"
if ($LASTEXITCODE -ne 0) { throw 'Cache download failed; laptop must remain connected.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $cacheZip).Hash.ToLowerInvariant() -ne $cacheMeta.sha256) {
    throw 'Cache archive checksum mismatch; training not started.'
}
$cacheMeta | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $WorkDirectory 'transfer-complete.json')
Write-Output 'TRANSFER VERIFIED. Windows now has all code/data and can continue without the laptop.'
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output "Workspace: $WorkDirectory"
Write-Output 'Keep Windows powered on. Results/logs remain in this workspace.'
