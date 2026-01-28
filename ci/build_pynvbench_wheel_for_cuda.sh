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

set -euxo pipefail

# Target script for `docker run` command in build_multi_cuda_wheel.sh
# This script builds a single wheel for the container's CUDA version
# The /workspace pathnames are hard-wired here.

# Determine CUDA version from nvcc early (needed for dev package installation)
cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
cuda_version_major=$(echo "${cuda_version}" | cut -d. -f1)
echo "Detected CUDA version: ${cuda_version}"

# Select CUDA architectures for multi-arch cubins + PTX fallback (if not set)
if [[ -z "${CUDAARCHS:-}" ]]; then
  version_ge() {
    [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
  }

  if version_ge "${cuda_version}" "13.0"; then
    CUDAARCHS="75-real;80-real;86-real;90a-real;100f-real;120a-real;120-virtual"
  elif version_ge "${cuda_version}" "12.9"; then
    CUDAARCHS="70-real;75-real;80-real;86-real;90a-real;100f-real;120a-real;120-virtual"
  else
    CUDAARCHS="70-real;75-real;80-real;86-real;90a-real;90-virtual"
    if version_ge "${cuda_version}" "12.8"; then
      CUDAARCHS="70-real;75-real;80-real;86-real;90a-real;100-real;120a-real;120-virtual"
    fi
  fi
fi
export CUDAARCHS
echo "Using CUDAARCHS: ${CUDAARCHS}"

# Install GCC 13 toolset (needed for the build)
/workspace/ci/util/retry.sh 5 30 dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh

# Note: CUDA dev packages (NVML, CUPTI, CUDART) are already installed in rapidsai/ci-wheel containers

# Set up Python environment
source /workspace/ci/pyenv_helper.sh
setup_python_env "${py_version}"
which python
python --version
echo "Done setting up python env"

# Ensure we have full git history for setuptools_scm
if $(git rev-parse --is-shallow-repository); then
  git fetch --unshallow
fi

cd /workspace/python

# Configure compilers
export CXX="$(which g++)"
export CUDACXX="$(which nvcc)"
export CUDAHOSTCXX="$(which g++)"

# Build the wheel
python -m pip wheel --no-deps --verbose --wheel-dir dist .

# Temporarily rename wheel to include CUDA version to avoid collision during multi-CUDA build
# The merge script will combine these into a single wheel
for wheel in dist/pynvbench-*.whl; do
    if [[ -f "$wheel" ]]; then
        base_name=$(basename "$wheel" .whl)
        new_name="${base_name}.cu${cuda_version_major}.whl"
        mv "$wheel" "dist/${new_name}"
        echo "Renamed wheel to: ${new_name}"
    fi
done

# Move wheel to output directory
mkdir -p /workspace/wheelhouse
mv dist/pynvbench-*.cu*.whl /workspace/wheelhouse/
