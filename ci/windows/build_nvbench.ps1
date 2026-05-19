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
    [string]$CMAKE_OPTIONS = "",

    [Parameter(Mandatory = $false)]
    [Alias("device-testing")]
    [bool]$DEVICE_TESTING = $false
)

$ErrorActionPreference = "Stop"

$initialPath = Get-Location
$pushed = $false

if ((Split-Path $pwd -Leaf) -ne "ci") {
    Push-Location "$PSScriptRoot/.."
    $pushed = $true
}

try {
    Import-Module "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS) -Force

    Print-EnvironmentDetails

    $preset = "nvbench-ci"
    $deviceTestingOption = if ($DEVICE_TESTING) { "ON" } else { "OFF" }
    $localOptions = @(
        "-DCMAKE_CXX_STANDARD=$CXX_STANDARD",
        "-DCMAKE_CUDA_STANDARD=$CXX_STANDARD",
        "-DNVBench_ENABLE_DEVICE_TESTING=$deviceTestingOption"
    )

    Configure-And-Build-Preset "NVBench" $preset $localOptions
} finally {
    if ($pushed) {
        Set-Location $initialPath
    }
}
