# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Param(
    [Parameter(Mandatory = $false)]
    [Alias("cudaVersion")]
    [string]$CUDA_VERSION = ""
)

$ErrorActionPreference = "Stop"

$RedistRootUri = "https://developer.download.nvidia.com/compute/cuda/redist"

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
            # missing package, or unsupported method. Keep 408/429 and 5xx on
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

function Read-JsonFile {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Path
    )

    try {
        $content = Get-Content -LiteralPath $Path -Raw
        $json = $content | ConvertFrom-Json
        return $json
    } catch {
        throw "Failed to parse JSON file '$Path'. $_"
    }
}

function Get-JsonPropertyValue {
    Param(
        [Parameter(Mandatory = $true)]
        $Object,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }

    $property = $Object.PSObject.Properties[$Name]
    if (-not $property) {
        return $null
    }

    return $property.Value
}

function Get-ComponentVersion {
    Param(
        [Parameter(Mandatory = $true)]
        $JsonObject,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$ComponentName
    )

    $component = Get-JsonPropertyValue -Object $JsonObject -Name $ComponentName
    if ($null -eq $component) {
        return ""
    }

    $version = Get-JsonPropertyValue -Object $component -Name "version"
    if ($null -eq $version) {
        return ""
    }

    return [string]$version
}

function Get-CudaVersionFromRoot {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$CudaRoot
    )

    $pathVersion = Get-CudaVersionFromPath -Path $CudaRoot
    if ($pathVersion) {
        return $pathVersion
    }

    $versionJson = Join-Path $CudaRoot "version.json"
    if (Test-Path $versionJson) {
        $versionData = Read-JsonFile -Path $versionJson
        $cudaVersion = Get-ComponentVersion -JsonObject $versionData -ComponentName "cuda"
        if ($cudaVersion -match '^(?<version>\d+\.\d+)(\.|$)') {
            return $Matches.version
        }
    }

    return ""
}

function Assert-Sha256 {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$ExpectedSha256
    )

    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedSha256 = $ExpectedSha256.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "SHA256 mismatch for '$Path'. Expected '$expectedSha256', got '$actualSha256'."
    }

    Write-Host "Validated SHA256 for '$Path': $actualSha256"
}

function Get-RedistribManifestNames {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$CudaVersionTag
    )

    $indexFile = Join-Path $env:TEMP "cuda_redist_index_$PID.html"
    try {
        Invoke-WebRequestWithRetry -Uri "$RedistRootUri/" -OutFile $indexFile
        $indexContent = Get-Content -LiteralPath $indexFile -Raw
    } finally {
        Remove-Item $indexFile -ErrorAction SilentlyContinue
    }

    $pattern = "redistrib_$([regex]::Escape($CudaVersionTag))\.\d+\.json"
    $manifestNames = @(
        [regex]::Matches($indexContent, $pattern) |
            ForEach-Object { $_.Value } |
            Sort-Object -Unique
    )

    if ($manifestNames.Count -eq 0) {
        throw "No CUDA $CudaVersionTag redistrib manifests were found at $RedistRootUri."
    }

    return @(
        $manifestNames |
            ForEach-Object {
                [PSCustomObject]@{
                    Name = $_
                    Version = [Version](($_ -replace '^redistrib_', '') -replace '\.json$', '')
                }
            } |
            Sort-Object -Property Version -Descending |
            ForEach-Object { $_.Name }
    )
}

function Read-RedistManifest {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$ManifestName
    )

    $manifestFile = Join-Path $env:TEMP $ManifestName
    try {
        Invoke-WebRequestWithRetry -Uri "$RedistRootUri/$ManifestName" -OutFile $manifestFile
        return Read-JsonFile -Path $manifestFile
    } finally {
        Remove-Item $manifestFile -ErrorAction SilentlyContinue
    }
}

function Select-ProfilerApiManifest {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$CudaVersionTag,

        [Parameter(Mandatory = $true)]
        $VersionData
    )

    $localProfilerApiVersion = Get-ComponentVersion `
        -JsonObject $VersionData `
        -ComponentName "cuda_profiler_api"
    $manifestNames = Get-RedistribManifestNames -CudaVersionTag $CudaVersionTag

    if ($localProfilerApiVersion) {
        Write-Host "CUDA version metadata reports cuda_profiler_api $localProfilerApiVersion."
    } else {
        Write-Host "CUDA version metadata does not report cuda_profiler_api; matching by installed core components."
    }

    $matchComponents = @("cuda_cupti", "cuda_cudart", "cuda_nvcc", "cuda_cccl")
    $bestCandidate = $null

    foreach ($manifestName in $manifestNames) {
        $manifest = Read-RedistManifest -ManifestName $manifestName
        $manifestProfilerApiVersion = Get-ComponentVersion `
            -JsonObject $manifest `
            -ComponentName "cuda_profiler_api"

        if (-not $manifestProfilerApiVersion) {
            continue
        }

        if ($localProfilerApiVersion) {
            if ($manifestProfilerApiVersion -eq $localProfilerApiVersion) {
                Write-Host "Selected CUDA redist manifest $manifestName."
                return [PSCustomObject]@{
                    Name = $manifestName
                    Manifest = $manifest
                }
            }
            continue
        }

        $matches = 0
        $mismatches = @()
        foreach ($componentName in $matchComponents) {
            $localVersion = Get-ComponentVersion `
                -JsonObject $VersionData `
                -ComponentName $componentName
            $manifestVersion = Get-ComponentVersion `
                -JsonObject $manifest `
                -ComponentName $componentName

            if (-not $localVersion -or -not $manifestVersion) {
                continue
            }

            if ($localVersion -eq $manifestVersion) {
                $matches++
            } else {
                $mismatches += "$componentName local=$localVersion manifest=$manifestVersion"
            }
        }

        if ($matches -gt 0 -and $mismatches.Count -eq 0) {
            if ($null -eq $bestCandidate -or $matches -gt $bestCandidate.MatchCount) {
                $bestCandidate = [PSCustomObject]@{
                    Name = $manifestName
                    Manifest = $manifest
                    MatchCount = $matches
                }
            }
        }
    }

    if ($localProfilerApiVersion) {
        throw "Could not find a CUDA $CudaVersionTag redistrib manifest with cuda_profiler_api $localProfilerApiVersion."
    }

    if ($null -eq $bestCandidate) {
        throw "Could not match installed CUDA Toolkit component versions to a CUDA $CudaVersionTag redistrib manifest."
    }

    Write-Host "Selected CUDA redist manifest $($bestCandidate.Name) using $($bestCandidate.MatchCount) component version match(es)."
    return [PSCustomObject]@{
        Name = $bestCandidate.Name
        Manifest = $bestCandidate.Manifest
    }
}

function Get-PayloadRoot {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$ExtractDir
    )

    $directories = @(Get-ChildItem -LiteralPath $ExtractDir -Directory)
    $files = @(Get-ChildItem -LiteralPath $ExtractDir -File)
    if ($directories.Count -eq 1 -and $files.Count -eq 0) {
        return $directories[0].FullName
    }

    return $ExtractDir
}

function Install-ProfilerApiPackage {
    Param(
        [Parameter(Mandatory = $true)]
        $ManifestSelection,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$CudaRoot
    )

    $component = Get-JsonPropertyValue `
        -Object $ManifestSelection.Manifest `
        -Name "cuda_profiler_api"
    if ($null -eq $component) {
        throw "Manifest $($ManifestSelection.Name) does not contain cuda_profiler_api."
    }

    $package = Get-JsonPropertyValue -Object $component -Name "windows-x86_64"
    if ($null -eq $package) {
        throw "Manifest $($ManifestSelection.Name) does not contain cuda_profiler_api for windows-x86_64."
    }

    $relativePath = Get-JsonPropertyValue -Object $package -Name "relative_path"
    $expectedSha256 = Get-JsonPropertyValue -Object $package -Name "sha256"
    if (-not $relativePath -or -not $expectedSha256) {
        throw "Manifest $($ManifestSelection.Name) is missing cuda_profiler_api relative_path or sha256."
    }
    if ($relativePath -notmatch '^cuda_profiler_api/windows-x86_64/cuda_profiler_api-windows-x86_64-[^/]+-archive\.zip$') {
        throw "Unexpected cuda_profiler_api package path in $($ManifestSelection.Name): $relativePath"
    }

    $pathParts = $relativePath -split '/'
    $archiveName = $pathParts[$pathParts.Length - 1]
    $archive = Join-Path $env:TEMP $archiveName
    $extractDir = Join-Path $env:TEMP "cuda_profiler_api_$([Guid]::NewGuid().ToString('N'))"
    $archiveUri = "$RedistRootUri/$relativePath"

    try {
        Write-Host "Downloading CUDA Profiler API redist package: $archiveUri"
        Invoke-WebRequestWithRetry -Uri $archiveUri -OutFile $archive
        Assert-Sha256 -Path $archive -ExpectedSha256 $expectedSha256

        Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
        $payloadRoot = Get-PayloadRoot -ExtractDir $extractDir
        $payloadHeader = Join-Path $payloadRoot "include\cuda_profiler_api.h"
        if (-not (Test-Path $payloadHeader)) {
            throw "CUDA Profiler API archive did not contain expected header: $payloadHeader"
        }

        Write-Host "Installing CUDA Profiler API package into: $CudaRoot"
        Copy-Item -Path (Join-Path $payloadRoot "*") -Destination $CudaRoot -Recurse -Force
    } finally {
        Remove-Item $archive -ErrorAction SilentlyContinue
        Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $CUDA_VERSION) {
    throw "CUDA Toolkit version is required. Provide -cudaVersion <major>.<minor>, for example '13.0'."
}

if ($CUDA_VERSION -notmatch '^\d+\.\d+$') {
    throw "Invalid CUDA Toolkit version '$CUDA_VERSION'. Expected '<major>.<minor>', for example '13.0'."
}

$version = [Version]$CUDA_VERSION
$mmVersionTag = "$($version.Major).$($version.Minor)"

$nvccCudaRoot = Get-CudaRootFromNvcc
if ($nvccCudaRoot) {
    $nvccCudaVersion = Get-CudaVersionFromRoot -CudaRoot $nvccCudaRoot
    if (-not $nvccCudaVersion) {
        throw "Could not determine CUDA version from active nvcc.exe root: $nvccCudaRoot"
    }
    if ($nvccCudaVersion -ne $mmVersionTag) {
        throw "Active nvcc.exe is from CUDA $nvccCudaVersion, but CUDA $mmVersionTag was requested."
    }
}

if ($env:CUDA_PATH) {
    $cudaPathVersion = Get-CudaVersionFromRoot -CudaRoot $env:CUDA_PATH
    if (-not $cudaPathVersion) {
        throw "Could not determine CUDA version from CUDA_PATH: $env:CUDA_PATH"
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

$profilerHeader = Join-Path $cudaRoot "include\cuda_profiler_api.h"
if (Test-Path $profilerHeader) {
    Write-Host "CUDA Profiler API is already installed: $profilerHeader"
    return
}

$versionJson = Join-Path $cudaRoot "version.json"
if (-not (Test-Path $versionJson)) {
    throw "CUDA Toolkit version metadata was not found: $versionJson. Cannot determine the matching cuda_profiler_api redist package."
}

$versionData = Read-JsonFile -Path $versionJson
$manifestSelection = Select-ProfilerApiManifest `
    -CudaVersionTag $mmVersionTag `
    -VersionData $versionData
Install-ProfilerApiPackage `
    -ManifestSelection $manifestSelection `
    -CudaRoot $cudaRoot

if (-not (Test-Path $profilerHeader)) {
    throw "CUDA Profiler API installation completed, but header was not found: $profilerHeader"
}

Write-Host "CUDA Profiler API installed: $profilerHeader"
