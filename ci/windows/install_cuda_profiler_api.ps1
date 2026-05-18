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

function Get-CudaRootFromNvcc {
    $nvccCommand = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
    if (-not $nvccCommand) {
        return ""
    }

    $nvccPath = $nvccCommand.Source
    $binDir = Split-Path -Parent $nvccPath
    if ((Split-Path -Leaf $binDir) -ne "bin") {
        throw "Could not derive CUDA root from nvcc.exe path: $nvccPath"
    }

    return Split-Path -Parent $binDir
}

function Assert-SamePath {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Left,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Right,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message
    )

    $leftFullPath = [System.IO.Path]::GetFullPath($Left).TrimEnd('\', '/')
    $rightFullPath = [System.IO.Path]::GetFullPath($Right).TrimEnd('\', '/')
    if ($leftFullPath -ne $rightFullPath) {
        throw "$Message Left='$leftFullPath' Right='$rightFullPath'"
    }
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

    $expectedPublisher = "NVIDIA Corporation"
    $publisher = $signature.SignerCertificate.GetNameInfo(
        [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,
        $false
    )
    if ($publisher -ne $expectedPublisher) {
        throw "Unexpected signer for '$Path': $publisher"
    }

    Write-Host "Validated Authenticode signature for '$Path': $publisher"
}

function Get-HttpStatusCodeFromError {
    Param(
        [Parameter(Mandatory = $true)]
        $ErrorRecord
    )

    $responseProperty = $ErrorRecord.Exception.PSObject.Properties["Response"]
    if (-not $responseProperty) {
        return $null
    }

    $response = $responseProperty.Value
    if ($null -eq $response) {
        return $null
    }

    $statusCodeProperty = $response.PSObject.Properties["StatusCode"]
    if (-not $statusCodeProperty) {
        return $null
    }

    return [int]$statusCodeProperty.Value
}

function Invoke-WebRequestWithRetry {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Uri,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$OutFile,

        [Parameter(Mandatory = $false)]
        [ValidateRange(1, 10)]
        [int]$MaxAttempts = 3
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            Remove-Item $OutFile -ErrorAction SilentlyContinue
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -TimeoutSec 300
            return
        } catch {
            $statusCode = Get-HttpStatusCodeFromError -ErrorRecord $_
            # Fail fast for deterministic client errors that indicate a bad URL,
            # missing installer, or unsupported method. Keep 408/429 and 5xx on
            # the retry path because they are commonly transient in CI.
            if (@(400, 401, 403, 404, 405, 410, 414) -contains $statusCode) {
                throw "Download failed with non-retryable HTTP status $statusCode from '$Uri'. $_"
            }

            if ($attempt -eq $MaxAttempts) {
                throw
            }

            $delaySeconds = 5 * $attempt
            Write-Warning "Download failed on attempt $attempt of $MaxAttempts. Retrying in $delaySeconds seconds. $_"
            Start-Sleep -Seconds $delaySeconds
        }
    }
}

if (-not $CUDA_VERSION) {
    $CUDA_VERSION = Get-CudaVersionFromPath -Path $env:CUDA_PATH
    if (-not $CUDA_VERSION) {
        throw "Could not determine CUDA version. Provide -cudaVersion or set CUDA_PATH to a path ending in v<major>.<minor>."
    }
}

if ($CUDA_VERSION -notmatch '^\d+\.\d+(\.\d+)?$') {
    throw "Invalid CUDA version '$CUDA_VERSION'. Expected '<major>.<minor>' or '<major>.<minor>.<patch>', for example '13.0' or '13.0.2'."
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

$nvccCudaRoot = Get-CudaRootFromNvcc
if ($nvccCudaRoot) {
    $nvccCudaVersion = Get-CudaVersionFromPath -Path $nvccCudaRoot
    if (-not $nvccCudaVersion) {
        throw "Could not determine CUDA version from active nvcc.exe root: $nvccCudaRoot"
    }
    if ($nvccCudaVersion -ne $mmVersionTag) {
        throw "Active nvcc.exe is from CUDA $nvccCudaVersion, but CUDA $mmVersionTag was requested."
    }
}

if ($env:CUDA_PATH) {
    $cudaPathVersion = Get-CudaVersionFromPath -Path $env:CUDA_PATH
    if (-not $cudaPathVersion) {
        throw "CUDA_PATH is set but does not end in v<major>.<minor>: $env:CUDA_PATH"
    }
    if ($cudaPathVersion -ne $mmVersionTag) {
        throw "CUDA_PATH points to CUDA $cudaPathVersion, but CUDA $mmVersionTag was requested."
    }
    if ($nvccCudaRoot) {
        Assert-SamePath `
            -Left $env:CUDA_PATH `
            -Right $nvccCudaRoot `
            -Message "CUDA_PATH and active nvcc.exe point to different CUDA Toolkit roots."
    }
    $cudaRoot = $env:CUDA_PATH
} elseif ($nvccCudaRoot) {
    $cudaRoot = $nvccCudaRoot
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
Invoke-WebRequestWithRetry -Uri $cudaVersionUrl -OutFile $installer
Assert-NvidiaAuthenticodeSignature -Path $installer

$installerTimeoutSeconds = 900
$process = $null
try {
    $process = Start-Process -PassThru -FilePath $installer -ArgumentList @("-s", $component)
    if (-not $process.WaitForExit($installerTimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "CUDA network installer timed out after $installerTimeoutSeconds seconds."
    }

    if ($process.ExitCode -ne 0) {
        throw "CUDA network installer failed with exit code $($process.ExitCode)."
    }
} finally {
    if ($process) {
        $process.Dispose()
    }
    Remove-Item $installer -ErrorAction SilentlyContinue
}

if (-not (Test-Path $profilerHeader)) {
    throw "CUDA Profiler API installation completed, but header was not found: $profilerHeader"
}

Write-Host "CUDA Profiler API installed: $profilerHeader"
