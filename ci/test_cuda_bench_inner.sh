#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in test_cuda_bench.sh
# The /workspace pathnames are hard-wired here.

# Install GCC 13 toolset (needed for builds that might happen during testing)
/workspace/ci/util/retry.sh 5 30 dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
# shellcheck source=/dev/null
source /etc/profile.d/enable_devtools.sh

: "${py_version:?py_version must be set}"
: "${cuda_version:?cuda_version must be set}"

# Set up Python environment.
# shellcheck source=ci/pyenv_helper.sh
source /workspace/ci/pyenv_helper.sh
setup_python_env "${py_version}"

# Upgrade pip
python -m pip install --upgrade pip

echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"

# Wheel should be in /workspace/wheelhouse (downloaded by workflow or built locally)
WHEELHOUSE_DIR="/workspace/wheelhouse"

# Find the cuda-bench wheel (multi-CUDA wheel)
# Prefer manylinux wheels, fall back to any wheel
CUDA_BENCH_WHEEL_PATH="$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name 'cuda_bench-*manylinux*.whl' -print -quit)"
if [[ -z "$CUDA_BENCH_WHEEL_PATH" ]]; then
    CUDA_BENCH_WHEEL_PATH="$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name 'cuda_bench-*.whl' -print -quit)"
fi

if [[ -z "$CUDA_BENCH_WHEEL_PATH" ]]; then
    echo "Error: No cuda-bench wheel found in ${WHEELHOUSE_DIR}"
    echo "Contents of ${WHEELHOUSE_DIR}:"
    ls -la ${WHEELHOUSE_DIR}/ || true
    exit 1
fi

# Determine which CUDA extra to install.
TEST_EXTRA="test-cu${cuda_version}"

echo "Installing wheel: $CUDA_BENCH_WHEEL_PATH with extras: ${TEST_EXTRA}"
python -m pip install "${CUDA_BENCH_WHEEL_PATH}[${TEST_EXTRA}]"

# Run tests
cd "/workspace/python/test/"
python -m pytest -v .
