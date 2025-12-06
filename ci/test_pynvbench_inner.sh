#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in test_pynvbench.sh
# The /workspace pathnames are hard-wired here.

# Install GCC 13 toolset (needed for builds that might happen during testing)
/workspace/ci/util/retry.sh 5 30 dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh

# Set up Python environment (only if not already available)
source /workspace/ci/pyenv_helper.sh
if ! command -v python${py_version} &> /dev/null; then
    setup_python_env "${py_version}"
fi

# Upgrade pip
python -m pip install --upgrade pip

echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"

# Wheel should be in /workspace/wheelhouse (downloaded by workflow or built locally)
WHEELHOUSE_DIR="/workspace/wheelhouse"

# Find the pynvbench wheel (multi-CUDA wheel)
# Prefer manylinux wheels, fall back to any wheel
PYNVBENCH_WHEEL_PATH="$(ls ${WHEELHOUSE_DIR}/pynvbench-*manylinux*.whl 2>/dev/null | head -1)"
if [[ -z "$PYNVBENCH_WHEEL_PATH" ]]; then
    PYNVBENCH_WHEEL_PATH="$(ls ${WHEELHOUSE_DIR}/pynvbench-*.whl 2>/dev/null | head -1)"
fi

if [[ -z "$PYNVBENCH_WHEEL_PATH" ]]; then
    echo "Error: No pynvbench wheel found in ${WHEELHOUSE_DIR}"
    echo "Contents of ${WHEELHOUSE_DIR}:"
    ls -la ${WHEELHOUSE_DIR}/ || true
    exit 1
fi

# Determine which CUDA extra to install (defaults to cu12 if not specified)
CUDA_EXTRA="${cuda_extra:-cu${cuda_version}}"
TEST_EXTRA="test-cu${cuda_version}"

echo "Installing wheel: $PYNVBENCH_WHEEL_PATH with extras: ${TEST_EXTRA}"
python -m pip install "${PYNVBENCH_WHEEL_PATH}[${TEST_EXTRA}]"

# Run tests
cd "/workspace/python/test/"
python -m pytest -v test_nvbench.py
