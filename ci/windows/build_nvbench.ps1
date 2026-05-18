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
    [Alias("run-tests")]
    [string]$RUN_TESTS = "false",

    [Parameter(Mandatory = $false)]
    [Alias("device-testing")]
    [string]$DEVICE_TESTING = "false"
)

$ErrorActionPreference = "Stop"

function ConvertTo-Bool {
    Param(
        [Parameter(Mandatory = $false)]
        [AllowNull()]
        [string]$Value = ""
    )

    $normalized = if ($null -eq $Value) { "" } else { $Value.Trim().ToLowerInvariant() }
    if (@("1", "true", "yes", "on") -contains $normalized) {
        return $true
    }
    if (@("0", "false", "no", "off", "") -contains $normalized) {
        return $false
    }

    throw "Expected a boolean-like value, got '$Value'."
}

$initialPath = Get-Location
$pushed = $false

if ((Split-Path $pwd -Leaf) -ne "ci") {
    Push-Location "$PSScriptRoot/.."
    $pushed = $true
}

try {
    Import-Module "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS) -Force

    $runTests = ConvertTo-Bool $RUN_TESTS
    $deviceTesting = ConvertTo-Bool $DEVICE_TESTING

    Print-EnvironmentDetails
    Write-Host "RUN_TESTS=$runTests"
    Write-Host "DEVICE_TESTING=$deviceTesting"

    $preset = "nvbench-ci"
    $localOptions = @(
        "-DCMAKE_CXX_STANDARD=$CXX_STANDARD",
        "-DCMAKE_CUDA_STANDARD=$CXX_STANDARD"
    )
    if ($deviceTesting) {
        $localOptions += "-DNVBench_ENABLE_DEVICE_TESTING=ON"
    } else {
        $localOptions += "-DNVBench_ENABLE_DEVICE_TESTING=OFF"
    }

    Configure-And-Build-Preset "NVBench" $preset $localOptions
    if ($runTests) {
        Test-Preset "NVBench" $preset
    }
} finally {
    if ($pushed) {
        Set-Location $initialPath
    }
}
