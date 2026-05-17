# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Param(
    [Parameter(Mandatory = $false)]
    [Alias("cudaVersion")]
    [string]$CUDA_VERSION = ""
)

$ErrorActionPreference = "Stop"

function Get-CudaVersionFromPath {
    Param(
        [Parameter(Mandatory = $false)]
        [string]$Path = ""
    )

    if ($Path -and $Path -match "v(?<version>\d+\.\d+)[\\/]?$") {
        return $Matches.version
    }

    return ""
}

function Assert-NvidiaAuthenticodeSignature {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Path
    )

    $signature = Get-AuthenticodeSignature -FilePath $Path
    if ($signature.Status -ne "Valid") {
        throw "Invalid Authenticode signature for '$Path': $($signature.Status) $($signature.StatusMessage)"
    }

    $subject = $signature.SignerCertificate.Subject
    if ($subject -notmatch "NVIDIA") {
        throw "Unexpected signer for '$Path': $subject"
    }

    Write-Host "Validated Authenticode signature for '$Path': $subject"
}

if (-not $CUDA_VERSION) {
    $CUDA_VERSION = Get-CudaVersionFromPath -Path $env:CUDA_PATH
    if (-not $CUDA_VERSION) {
        throw "Could not determine CUDA version. Provide -cudaVersion or set CUDA_PATH to a path ending in v<major>.<minor>."
    }
}

$version = [Version]$CUDA_VERSION
$major = $version.Major
$minor = $version.Minor
$build = $version.Build

if ($build -lt 0) {
    $build = 0
}

$mmbVersionTag = "${major}.${minor}.${build}"
$mmVersionTag = "${major}.${minor}"

if ($env:CUDA_PATH) {
    $cudaPathVersion = Get-CudaVersionFromPath -Path $env:CUDA_PATH
    if (-not $cudaPathVersion) {
        throw "CUDA_PATH is set but does not end in v<major>.<minor>: $env:CUDA_PATH"
    }
    if ($cudaPathVersion -ne $mmVersionTag) {
        throw "CUDA_PATH points to CUDA $cudaPathVersion, but CUDA $mmVersionTag was requested."
    }
    $cudaRoot = $env:CUDA_PATH
} else {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$mmVersionTag"
}
$profilerHeader = "$cudaRoot\include\cuda_profiler_api.h"

if (Test-Path $profilerHeader) {
    Write-Host "CUDA Profiler API is already installed: $profilerHeader"
    return
}

$component = "cuda_profiler_api_$mmVersionTag"
$cudaMajorUri = "${mmbVersionTag}/network_installers/cuda_${mmbVersionTag}_windows_network.exe"
$cudaVersionUrl = "https://developer.download.nvidia.com/compute/cuda/$cudaMajorUri"
$installer = Join-Path $env:TEMP "cuda_${mmbVersionTag}_windows_network.exe"

Write-Host "Installing CUDA component: $component"
Write-Host "Downloading CUDA network installer: $cudaVersionUrl"
Invoke-WebRequest -Uri $cudaVersionUrl -OutFile $installer -UseBasicParsing
Assert-NvidiaAuthenticodeSignature -Path $installer

try {
    $process = Start-Process -Wait -PassThru -FilePath $installer -ArgumentList @("-s", $component)
    if ($process.ExitCode -ne 0) {
        throw "CUDA network installer failed with exit code $($process.ExitCode)."
    }
} finally {
    Remove-Item $installer -ErrorAction SilentlyContinue
}

if (-not (Test-Path $profilerHeader)) {
    throw "CUDA Profiler API installation completed, but header was not found: $profilerHeader"
}

Write-Host "CUDA Profiler API installed: $profilerHeader"
