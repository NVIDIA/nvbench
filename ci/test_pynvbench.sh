#!/bin/bash

set -euo pipefail
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

# Setup Python environment
setup_python_env "${py_version}"

# Fetch the pynvbench wheel
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # In GitHub Actions, wheel is already downloaded to wheelhouse/ by the workflow
  WHEELHOUSE_DIR="/workspace/wheelhouse"
else
  # For local testing, build the wheel
  "$ci_dir/build_pynvbench_wheel.sh" -py-version "${py_version}" -cuda-version "${cuda_version}"
  WHEELHOUSE_DIR="/workspace/wheelhouse"
fi

# Find and install pynvbench wheel
PYNVBENCH_WHEEL_PATH="$(ls ${WHEELHOUSE_DIR}/pynvbench-*+cu${cuda_version}*.whl 2>/dev/null | head -1)"
if [[ -z "$PYNVBENCH_WHEEL_PATH" ]]; then
    echo "Error: No pynvbench wheel found in ${WHEELHOUSE_DIR}"
    ls -la ${WHEELHOUSE_DIR}/ || true
    exit 1
fi

echo "Installing wheel: $PYNVBENCH_WHEEL_PATH"
python -m pip install "${PYNVBENCH_WHEEL_PATH}[test]"

# Run tests
cd "/workspace/python/test/"
python -m pytest -v test_nvbench.py
