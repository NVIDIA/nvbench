Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17, 20)]
    [int]$CXX_STANDARD = 17,

    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = "",

    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [string]$CMAKE_OPTIONS = ""
)

$ErrorActionPreference = "Stop"

function Split-CMakeOptions {
    Param(
        [Parameter(Mandatory = $false)]
        [string]$Options = ""
    )

    if (-not $Options) {
        return @()
    }

    return @($Options -split '\s+' | Where-Object { $_ })
}

function Invoke-OptionalSccache {
    Param(
        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    if (-not (Get-Command sccache -ErrorAction SilentlyContinue)) {
        return
    }

    & sccache @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "sccache $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

# Use the full cl.exe path. CMake otherwise may resolve CMAKE_CXX_COMPILER to a
# full path while leaving CMAKE_CUDA_HOST_COMPILER as "cl", which fails NVBench's
# host/compiler consistency check.
$script:HOST_COMPILER = (Get-Command "cl.exe").Source -replace '\\', '/'
$script:CUDA_COMPILER = (Get-Command "nvcc.exe").Source -replace '\\', '/'
$script:PARALLEL_LEVEL = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { 1 }

$script:GLOBAL_CMAKE_OPTIONS = @(Split-CMakeOptions $CMAKE_OPTIONS)
if ($CUDA_ARCH) {
    $script:GLOBAL_CMAKE_OPTIONS += "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
}

if (-not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = ""
}

$env:CMAKE_BUILD_PARALLEL_LEVEL = $script:PARALLEL_LEVEL
$env:CTEST_PARALLEL_LEVEL = 1
$env:CMAKE_GENERATOR = "Ninja"
$env:CXX = $script:HOST_COMPILER
$env:CUDACXX = $script:CUDA_COMPILER
$env:CUDAHOSTCXX = $script:HOST_COMPILER

Invoke-OptionalSccache -Arguments @("--start-server")

function Print-EnvironmentDetails {
    Write-Host "========================================"
    Write-Host "Begin build"
    Write-Host "pwd=$pwd"
    Write-Host "CXX_STANDARD=$CXX_STANDARD"
    Write-Host "CXX=$env:CXX"
    Write-Host "CUDACXX=$env:CUDACXX"
    Write-Host "CUDAHOSTCXX=$env:CUDAHOSTCXX"
    Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"
    Write-Host "CTEST_PARALLEL_LEVEL=$env:CTEST_PARALLEL_LEVEL"
    Write-Host "CCCL_BUILD_INFIX=$env:CCCL_BUILD_INFIX"
    Write-Host "GLOBAL_CMAKE_OPTIONS=$($script:GLOBAL_CMAKE_OPTIONS -join ' ')"
    Write-Host "Current commit is:"
    git log -1 --format=short
    Write-Host "========================================"

    cmake --version
    ctest --version
    ninja --version
    cl.exe /?
    nvcc --version
    Invoke-OptionalSccache -Arguments @("--version")
}

function Invoke-NativeCommand {
    Param(
        [Parameter(Mandatory = $true)]
        [string]$Step,

        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    Write-Host ">>> $Step"
    Write-Host "$Command $($Arguments -join ' ')"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Configure-Preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,

        [Parameter(Mandatory = $false)]
        [string[]]$LOCAL_CMAKE_OPTIONS = @()
    )

    Push-Location ".."
    try {
        $args = @("--preset", $PRESET, "--log-level=VERBOSE")
        $args += $LOCAL_CMAKE_OPTIONS
        $args += $script:GLOBAL_CMAKE_OPTIONS
        Invoke-NativeCommand "$BUILD_NAME configure" "cmake" $args
    } finally {
        Pop-Location
    }
}

function Build-Preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    Push-Location ".."
    try {
        Invoke-OptionalSccache -Arguments @("-z")
        Invoke-NativeCommand "$BUILD_NAME build" "cmake" @("--build", "--preset=$PRESET", "-v")
        Invoke-OptionalSccache -Arguments @("--show-adv-stats")
    } finally {
        Pop-Location
    }
}

function Configure-And-Build-Preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,

        [Parameter(Mandatory = $false)]
        [string[]]$LOCAL_CMAKE_OPTIONS = @()
    )

    Configure-Preset $BUILD_NAME $PRESET $LOCAL_CMAKE_OPTIONS
    Build-Preset $BUILD_NAME $PRESET
}

function Test-Preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    Push-Location ".."
    try {
        Invoke-NativeCommand "$BUILD_NAME test" "ctest" @("--preset=$PRESET", "--output-on-failure")
    } finally {
        Pop-Location
    }
}

Export-ModuleMember -Function Print-EnvironmentDetails, Configure-Preset, Build-Preset, Configure-And-Build-Preset, Test-Preset
