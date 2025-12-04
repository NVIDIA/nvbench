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

# Install GCC 13 toolset (needed for the build)
/workspace/ci/util/retry.sh 5 30 dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh

# Check what's available
which gcc
gcc --version
which nvcc
nvcc --version

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

# Determine CUDA version from nvcc
cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | cut -d. -f1)
echo "Detected CUDA version: ${cuda_version}"

# Configure compilers
export CXX="$(which g++)"
export CUDACXX="$(which nvcc)"
export CUDAHOSTCXX="$(which g++)"

# Set CUDA suffix for extension naming
export PYNVBENCH_CUDA_SUFFIX="_cu${cuda_version}"

# Build the wheel
python -m pip wheel --no-deps --verbose --wheel-dir dist .

# Rename wheel to include CUDA version suffix
for wheel in dist/pynvbench-*.whl; do
    if [[ -f "$wheel" ]]; then
        base_name=$(basename "$wheel" .whl)
        new_name="${base_name}.cu${cuda_version}.whl"
        mv "$wheel" "dist/${new_name}"
        echo "Renamed wheel to: ${new_name}"
    fi
done

# Move wheel to output directory
mkdir -p /workspace/wheelhouse
mv dist/pynvbench-*.cu*.whl /workspace/wheelhouse/
