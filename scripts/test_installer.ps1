param([Parameter(Mandatory=$true)][string]$Installer)
$ErrorActionPreference = 'Stop'
if ($env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'Run this install/uninstall smoke check on a disposable GitHub-hosted Windows runner.'
}
$installDir = Join-Path ([System.IO.Path]::GetTempPath()) ('BlurAll-Install-Test-' + [guid]::NewGuid().ToString('N'))
$savedQtPlatform = $env:QT_QPA_PLATFORM

function Wait-CheckedProcess($Process, [int]$Timeout = 120000) {
    if (-not $Process.WaitForExit($Timeout)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw 'Installer test process timed out'
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0) { throw "Installer test process failed: $($Process.ExitCode)" }
}

try {
    $setup = Start-Process -FilePath (Resolve-Path $Installer).Path -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=`"$installDir`"" -PassThru
    Wait-CheckedProcess $setup
    $exe = Join-Path $installDir 'BlurAll.exe'
    if (-not (Test-Path $exe)) { throw 'Installed executable is missing' }
    if (-not (Test-Path (Join-Path $installDir 'third-party-licenses'))) { throw 'Dependency notices are missing' }
    $env:QT_QPA_PLATFORM = 'offscreen'
    $app = Start-Process -FilePath $exe -ArgumentList '--smoke-test' -PassThru
    Wait-CheckedProcess $app 60000
    Write-Output 'Installed application completed a real video export.'
}
finally {
    $env:QT_QPA_PLATFORM = $savedQtPlatform
    $uninstaller = Join-Path $installDir 'unins000.exe'
    if (Test-Path $uninstaller) {
        $uninstall = Start-Process -FilePath $uninstaller -ArgumentList '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -PassThru
        Wait-CheckedProcess $uninstall
        if (Test-Path (Join-Path $installDir 'BlurAll.exe')) { throw 'Uninstall left the application executable behind' }
    }
    if (Test-Path $installDir) { Remove-Item -LiteralPath $installDir -Recurse -Force }
}
Write-Output 'Install, launch, export, and uninstall checks passed.'
