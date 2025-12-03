#!/bin/bash

set -euo pipefail

# Enable verbose output for debugging
set -x
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Parse CUDA version
cuda_version=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -cuda-version=*)
            cuda_version="${1#*=}"
            shift
            ;;
        -cuda-version)
            if [[ $# -lt 2 ]]; then
                echo "Error: -cuda-version requires a value" >&2
                exit 1
            fi
            cuda_version="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$cuda_version" ]]; then
    echo "Error: -cuda-version is required"
    exit 1
fi

# Determine CUDA major version from environment
cuda_major_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1 | cut -d 'V' -f 2)

# Setup Python environment (skip if we're already in ci-wheel container with correct Python)
echo "Checking for Python ${py_version}..."
if command -v python &> /dev/null; then
    actual_py_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    echo "Found Python version: ${actual_py_version}"
    if [[ "${actual_py_version}" == "${py_version}" ]]; then
        echo "Python ${py_version} already available, skipping pyenv setup"
        python -m pip install --upgrade pip
    else
        echo "Python version mismatch (found ${actual_py_version}, need ${py_version})"
        echo "Setting up Python ${py_version} with pyenv"
        setup_python_env "${py_version}"
    fi
else
    echo "Python not found, setting up with pyenv"
    setup_python_env "${py_version}"
fi

echo "Python setup complete, version: $(python --version)"

# Wheel should be in /workspace/wheelhouse (downloaded by workflow or built locally)
WHEELHOUSE_DIR="/workspace/wheelhouse"

# Find and install pynvbench wheel
# Look for .cu${cuda_version} in the version string (e.g., pynvbench-0.0.1.dev1+g123.cu12-...)
PYNVBENCH_WHEEL_PATH="$(ls ${WHEELHOUSE_DIR}/pynvbench-*.cu${cuda_version}-*.whl 2>/dev/null | head -1)"
if [[ -z "$PYNVBENCH_WHEEL_PATH" ]]; then
    echo "Error: No pynvbench wheel found in ${WHEELHOUSE_DIR}"
    echo "Looking for: pynvbench-*.cu${cuda_version}-*.whl"
    echo "Contents of ${WHEELHOUSE_DIR}:"
    ls -la ${WHEELHOUSE_DIR}/ || true
    exit 1
fi

echo "Installing wheel: $PYNVBENCH_WHEEL_PATH"
python -m pip install "${PYNVBENCH_WHEEL_PATH}[test]"

# Run tests
cd "/workspace/python/test/"
python -m pytest -v test_nvbench.py
