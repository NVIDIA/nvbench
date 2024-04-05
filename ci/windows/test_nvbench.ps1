
Param(
    [Parameter(Mandatory = $true)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17)]
    [int]$CXX_STANDARD = 17
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD

$PRESET = "nvbench-cpp$CXX_STANDARD"
$CMAKE_OPTIONS = ""

configure_and_build_preset "NVBench" "$PRESET" "$CMAKE_OPTIONS"
test_preset "NVBench" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
