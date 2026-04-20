param(
    [string]$Config = "Release",
    [switch]$BuildIfMissing
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$exe = Join-Path $repoRoot "build/$Config/GravitySimulation.exe"

if (-not (Test-Path $exe)) {
    if ($BuildIfMissing) {
        Write-Host "Executable missing. Building first..."
        & "$repoRoot/build.ps1" -Config $Config
    } else {
        throw "Executable not found at $exe. Run .\build.ps1 -Config $Config first, or use .\run.ps1 -BuildIfMissing."
    }
}

if (-not (Test-Path $exe)) {
    throw "Executable still not found at $exe after build."
}

Write-Host "Running $exe ..."
& $exe
