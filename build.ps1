param(
    [switch]$Run,
    [switch]$SetupDeps,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$vcpkgRoot = Join-Path $env:USERPROFILE "vcpkg"
$toolchainFile = Join-Path $vcpkgRoot "scripts/buildsystems/vcpkg.cmake"
$buildDir = Join-Path $repoRoot "build"

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-BasicTools {
    if (-not (Test-CommandExists "cmake")) {
        throw "cmake was not found in PATH. Install CMake first (winget install Kitware.CMake)."
    }
    if (-not (Test-CommandExists "git")) {
        throw "git was not found in PATH. Install Git first."
    }
}

function Setup-Dependencies {
    Ensure-BasicTools

    if (-not (Test-Path $vcpkgRoot)) {
        Write-Host "Cloning vcpkg to $vcpkgRoot ..."
        git clone https://github.com/microsoft/vcpkg "$vcpkgRoot"
    } else {
        Write-Host "vcpkg already exists at $vcpkgRoot"
    }

    $bootstrap = Join-Path $vcpkgRoot "bootstrap-vcpkg.bat"
    if (-not (Test-Path $bootstrap)) {
        throw "vcpkg bootstrap file not found at $bootstrap"
    }

    Write-Host "Bootstrapping vcpkg ..."
    & $bootstrap

    $vcpkgExe = Join-Path $vcpkgRoot "vcpkg.exe"
    if (-not (Test-Path $vcpkgExe)) {
        throw "vcpkg executable not found at $vcpkgExe"
    }

    Write-Host "Installing dependencies: glfw3 glew glm ..."
    & $vcpkgExe install glfw3 glew glm
}

Ensure-BasicTools

if ($SetupDeps) {
    Setup-Dependencies
}

if (-not (Test-Path $toolchainFile)) {
    throw "vcpkg toolchain file not found: $toolchainFile. Run .\build.ps1 -SetupDeps first."
}

Write-Host "Configuring project ..."
cmake -S . -B "$buildDir" -DCMAKE_TOOLCHAIN_FILE="$toolchainFile"

Write-Host "Building project ($Config) ..."
cmake --build "$buildDir" --config "$Config"

if ($Run) {
    $exe = Join-Path $buildDir "$Config/GravitySimulation.exe"
    if (-not (Test-Path $exe)) {
        throw "Executable not found at $exe"
    }
    Write-Host "Running $exe ..."
    & $exe
}

Write-Host "Done."
