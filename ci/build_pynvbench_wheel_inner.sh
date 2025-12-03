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

# Install auditwheel for manylinux compliance
python -m pip install auditwheel

# Repair wheel to make it manylinux compliant
mkdir -p dist_repaired
for wheel in dist/pynvbench-*.whl; do
    if [[ -f "$wheel" ]]; then
        echo "Repairing wheel: $wheel"
        python -m auditwheel repair \
            --exclude 'libcuda.so.1' \
            --exclude 'libnvidia-ml.so.1' \
            "$wheel" \
            --wheel-dir dist_repaired
    fi
done

# Rename wheel to include CUDA version suffix
mkdir -p /workspace/wheelhouse
for wheel in dist_repaired/pynvbench-*.whl; do
    if [[ -f "$wheel" ]]; then
        base_name=$(basename "$wheel" .whl)
        # Insert CUDA version before the platform tag
        # e.g., pynvbench-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
        # becomes pynvbench-0.1.0+cu12-cp312-cp312-manylinux_2_28_x86_64.whl
        if [[ "$base_name" =~ ^(.*)-cp([0-9]+)-cp([0-9]+)-(.*) ]]; then
            pkg_version="${BASH_REMATCH[1]}"
            py_tag="cp${BASH_REMATCH[2]}"
            abi_tag="cp${BASH_REMATCH[3]}"
            platform="${BASH_REMATCH[4]}"
            new_name="${pkg_version}+cu${cuda_major}-${py_tag}-${abi_tag}-${platform}.whl"
            mv "$wheel" "/workspace/wheelhouse/${new_name}"
            echo "Renamed wheel to: ${new_name}"
        else
            # Fallback if regex doesn't match
            mv "$wheel" /workspace/wheelhouse/
            echo "Moved wheel: $(basename $wheel)"
        fi
    fi
done
