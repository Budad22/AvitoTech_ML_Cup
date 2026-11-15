# Temporary sleep prevention for the complete autonomous pipeline, including setup.
Add-Type 'public static class AvitoPower { [System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
try {
    [AvitoPower]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    do {
        Start-Sleep -Seconds 30
    } while ((Get-ScheduledTask -TaskName AvitoMLP).State -eq 'Running' -or
             ((Test-Path 'Q:\AVITOCUP\migration-state.json') -and !(Test-Path 'Q:\AVITOCUP\migration-release.flag')))
} finally {
    [AvitoPower]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
