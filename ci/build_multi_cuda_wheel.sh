#!/bin/bash
# Apache 2.0 License
# Copyright 2024-2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 with the LLVM exception
# (the "License"); you may not use this file except in compliance with
# the License.
#
# You may obtain a copy of the License at
#
#     http://llvm.org/foundation/relicensing/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build a multi-CUDA wheel for the given Python version
# This builds separate wheels for each supported CUDA major version,
# and then merges them into a single wheel containing extensions
# for all CUDA versions. At runtime, depending on the installed CUDA version,
# the correct extension will be chosen.

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Check if py_version was provided (this script requires it)
require_py_version "$usage" || exit 1

echo "Docker socket: " $(ls /var/run/docker.sock)

# Set HOST_WORKSPACE if not already set (for local runs)
if [[ -z "${HOST_WORKSPACE:-}" ]]; then
  # Get the repository root
  HOST_WORKSPACE="$(cd "${ci_dir}/.." && pwd)"
  echo "Setting HOST_WORKSPACE to: $HOST_WORKSPACE"
fi

# cuda-bench must be built in a container that can produce manylinux wheels,
# and has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# We build separate wheels using separate containers for each CUDA version,
# then merge them into a single wheel.

readonly cuda12_version=12.9.1
readonly cuda13_version=13.0.1
readonly devcontainer_version=25.12
readonly devcontainer_distro=rockylinux8

if [[ "$(uname -m)" == "aarch64" ]]; then
  readonly host_arch_suffix="-arm64"
else
  readonly host_arch_suffix=""
fi

readonly cuda12_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda12_version}-${devcontainer_distro}-py${py_version}${host_arch_suffix}
readonly cuda13_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda13_version}-${devcontainer_distro}-py${py_version}${host_arch_suffix}

mkdir -p wheelhouse

for ctk in 12 13; do
  image=$(eval echo \$cuda${ctk}_image)
  echo "::group::⚒️ Building CUDA ${ctk} wheel on ${image}"
  (
    set -x
    docker pull $image
    docker run --rm -i \
        --workdir /workspace/python \
        --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
        --env py_version=${py_version} \
        $image \
        /workspace/ci/build_cuda_bench_wheel_for_cuda.sh
    # Prevent GHA runners from exhausting available storage with leftover images:
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      docker rmi -f $image
    fi
  )
  echo "::endgroup::"
done

echo "Merging CUDA wheels..."

# Detect python command
if command -v python &> /dev/null; then
  PYTHON=python
elif command -v python3 &> /dev/null; then
  PYTHON=python3
else
  echo "Error: No python found"
  exit 1
fi

# Needed for unpacking and repacking wheels.
$PYTHON -m pip install --break-system-packages wheel

# Find the built wheels (temporarily suffixed with .cu12/.cu13 to avoid collision)
cu12_wheel=$(find wheelhouse -name "*cu12*.whl" | head -1)
cu13_wheel=$(find wheelhouse -name "*cu13*.whl" | head -1)

if [[ -z "$cu12_wheel" ]]; then
  echo "Error: CUDA 12 wheel not found in wheelhouse/"
  ls -la wheelhouse/
  exit 1
fi

if [[ -z "$cu13_wheel" ]]; then
  echo "Error: CUDA 13 wheel not found in wheelhouse/"
  ls -la wheelhouse/
  exit 1
fi

if [[ "$cu12_wheel" == "$cu13_wheel" ]]; then
  echo "Error: Only one wheel found, expected two (CUDA 12 and CUDA 13)"
  ls -la wheelhouse/
  exit 1
fi

echo "Found CUDA 12 wheel: $cu12_wheel"
echo "Found CUDA 13 wheel: $cu13_wheel"

# Convert to absolute paths before changing directory
cu12_wheel=$(readlink -f "$cu12_wheel")
cu13_wheel=$(readlink -f "$cu13_wheel")

# Merge the wheels manually
mkdir -p wheelhouse_merged
cd wheelhouse_merged

# Unpack CUDA 12 wheel (this will be our base)
$PYTHON -m wheel unpack "$cu12_wheel"
base_dir=$(find . -maxdepth 1 -type d -name "cuda-bench-*" | head -1)

# Unpack CUDA 13 wheel into a temporary subdirectory
mkdir cu13_tmp
cd cu13_tmp
$PYTHON -m wheel unpack "$cu13_wheel"
cu13_dir=$(find . -maxdepth 1 -type d -name "cuda-bench-*" | head -1)

# Copy the cu13/ directory from CUDA 13 wheel into the base wheel
cp -r "$cu13_dir"/cuda/bench/cu13 "../$base_dir/cuda/bench/"

# Go back and clean up
cd ..
rm -rf cu13_tmp

# Remove RECORD file to let wheel recreate it
rm -f "$base_dir"/*.dist-info/RECORD

# Repack the merged wheel
$PYTHON -m wheel pack "$base_dir"

cd ..

# Install auditwheel and repair the merged wheel
$PYTHON -m pip install --break-system-packages auditwheel
for wheel in wheelhouse_merged/cuda-bench-*.whl; do
    echo "Repairing merged wheel: $wheel"
    $PYTHON -m auditwheel repair \
        --exclude 'libcuda.so.1' \
        --exclude 'libnvidia-ml.so.1' \
        --exclude 'libcupti.so.12' \
        --exclude 'libcupti.so.13' \
        --exclude 'libnvperf_host.so' \
        --exclude 'libnvperf_target.so' \
        "$wheel" \
        --wheel-dir wheelhouse_final
done

# Clean up intermediate files and move only the final merged wheel to wheelhouse
rm -rf wheelhouse/*  # Clean existing wheelhouse
mkdir -p wheelhouse

# Move only the final repaired merged wheel
if ls wheelhouse_final/cuda-bench-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_final/cuda-bench-*.whl wheelhouse/
    echo "Final merged wheel moved to wheelhouse"
else
    echo "No final repaired wheel found, moving unrepaired merged wheel"
    mv wheelhouse_merged/cuda-bench-*.whl wheelhouse/
fi

# Clean up temporary directories
rm -rf wheelhouse_merged wheelhouse_final

echo "Final wheels in wheelhouse:"
ls -la wheelhouse/
