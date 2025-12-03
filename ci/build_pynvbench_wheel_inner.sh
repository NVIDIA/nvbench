#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in build_pynvbench_wheel.sh
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
cuda_major=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | cut -d. -f1)
echo "Detected CUDA major version: ${cuda_major}"

# Configure compilers:
export CXX="$(which g++)"
export CUDACXX="$(which nvcc)"
export CUDAHOSTCXX="$(which g++)"

# Build the wheel
python -m pip wheel --no-deps --verbose --wheel-dir dist .

# Rename wheel to include CUDA version suffix
for wheel in dist/pynvbench-*.whl; do
    if [[ -f "$wheel" ]]; then
        base_name=$(basename "$wheel" .whl)
        new_name="${base_name}+cu${cuda_major}-py${py_version//.}-linux_$(uname -m).whl"
        mv "$wheel" "dist/${new_name}"
        echo "Renamed wheel to: ${new_name}"
    fi
done

# Move wheel to output directory
mkdir -p /workspace/wheelhouse
mv dist/pynvbench-*+cu*.whl /workspace/wheelhouse/
