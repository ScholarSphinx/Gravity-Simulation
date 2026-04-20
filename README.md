# Gravity Simulation
Using OpenGL and C++ to simulate gravity in space

## Features
- 3D OpenGL rendering of a solar-system-like N-body simulation
- Newtonian gravity with a relativity-inspired correction term
- Inelastic collision handling (bodies merge and conserve momentum)
- White 2D spacetime grid rendered as a dynamic paraboloid warp surface
- Planet orbit trails to visualize orbital paths
- Per-pixel lighting/shading on spherical celestial bodies with procedural textures
- Seeded solar system bodies through Pluto, including Earth's Moon
- Runtime spawning with dynamic names:
  - `P`: spawn `PLANET #x`
  - `B`: spawn `BLACK HOLE #x`
- Per-body labels above each object
- Camera controls for orbit/pan/zoom and reset

## Build Requirements
- CMake 3.20+
- C++17 compiler
- OpenGL
- GLFW
- GLEW
- GLM

## Build
```bash
cmake -S . -B build
cmake --build build --config Release
```

## One-command script (Windows)
You can use the included script to configure/build, and optionally run:

```powershell
# First-time setup + build + run
.\build.ps1 -SetupDeps -Run

# Normal rebuild + run
.\build.ps1 -Run

# Build only
.\build.ps1
```

## Quick run script (Windows)
Use `run.ps1` to launch the built executable directly:

```powershell
# Run Release build
.\run.ps1

# Run Debug build
.\run.ps1 -Config Debug

# Auto-build if executable is missing, then run
.\run.ps1 -BuildIfMissing
```

## Double-click launcher (Windows)
Use `start-sim.bat` to run the simulation without calling PowerShell scripts directly.

- If `build\Release\GravitySimulation.exe` exists, it launches immediately.
- If missing, it builds via `build.ps1` using PowerShell bypass, then launches.

You can run it by double-clicking `start-sim.bat` in File Explorer, or from terminal:

```bat
start-sim.bat
```

## PowerShell: "running scripts is disabled"
If PowerShell blocks `.ps1` files, use one of these options.

One-time bypass for a single command (no permanent policy change):

```powershell
powershell -ExecutionPolicy Bypass -File ".\build.ps1" -SetupDeps -Run
powershell -ExecutionPolicy Bypass -File ".\run.ps1"
```

Session-only bypass (resets when terminal closes):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build.ps1 -SetupDeps -Run
```

Persistent per-user setting (optional):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Run
- On single-config generators:
  - `./build/GravitySimulation`
- On Visual Studio (multi-config):
  - `./build/Release/GravitySimulation.exe`

## Controls
- Simulation:
  - `P`: spawn a random planet
  - `B`: spawn a black hole
  - `Space`: pause/resume
  - `Esc`: quit
- Camera:
  - Mouse left drag: orbit camera
  - Mouse right drag: pan camera
  - Mouse wheel: zoom (and adjust FOV)
  - `+` / `-`: zoom in/out
  - `W`, `A`, `S`, `D`: pan forward/left/back/right
  - `Q` / `E`: pan up/down
  - `R`: reset camera

## Windows Setup (if tools are missing)
If `cmake` is not recognized, install the build tools first:

```powershell
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools
```

Then install C++ libraries with vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg "$env:USERPROFILE\vcpkg"
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat"
& "$env:USERPROFILE\vcpkg\vcpkg.exe" install glfw3 glew glm
```

Configure using the vcpkg toolchain:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
```

## Notes on Physics Model
- Gravity is computed using pairwise Newtonian attraction.
- A simplified relativistic correction is applied to increase effective gravity in strong/high-speed regimes.
- Spacetime curvature is visualized using a white paraboloid grid deformed by mass potential.
- This is a visually grounded approximation of GR behavior for interactive simulation, not a full tensor-based Einstein field solver.
