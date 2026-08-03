#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in test_python_examples.sh.
# The /workspace pathnames are hard-wired here.

: "${py_version:?py_version must be set}"
: "${cuda_extra:?cuda_extra must be set}"

# shellcheck source=ci/pyenv_helper.sh
source /workspace/ci/pyenv_helper.sh

WHEELHOUSE_DIR="/workspace/wheelhouse"
EXAMPLE_REQUIREMENTS_DIR="/workspace/python/examples/requirements"

if [[ ! -d "${WHEELHOUSE_DIR}" ]]; then
    echo "Error: Wheelhouse directory not found: ${WHEELHOUSE_DIR}" >&2
    echo "Build or download the cuda-bench wheel into wheelhouse before running example tests." >&2
    exit 1
fi

CUDA_BENCH_WHEEL_PATH="$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name 'cuda_bench-*manylinux*.whl' -print -quit)"
if [[ -z "$CUDA_BENCH_WHEEL_PATH" ]]; then
    CUDA_BENCH_WHEEL_PATH="$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name 'cuda_bench-*.whl' -print -quit)"
fi

if [[ -z "$CUDA_BENCH_WHEEL_PATH" ]]; then
    echo "Error: No cuda-bench wheel found in ${WHEELHOUSE_DIR}" >&2
    echo "Contents of ${WHEELHOUSE_DIR}:" >&2
    ls -la "${WHEELHOUSE_DIR}/" || true
    exit 1
fi

run_example_env() {
    local env_name="$1"
    shift
    local group_spec="$1"
    shift
    local requirements_file="${EXAMPLE_REQUIREMENTS_DIR}/${env_name}.txt"
    local groups=()
    local runner_args=("--repo-root" "/workspace" "--continue-on-failure")

    if [[ ! -f "${requirements_file}" ]]; then
        echo "Error: Missing requirements file: ${requirements_file}" >&2
        exit 1
    fi

    IFS=',' read -r -a groups <<< "${group_spec}"
    for group in "${groups[@]}"; do
        runner_args+=("--group" "${group}")
    done

    export NVBENCH_PYTHON_VENV="${HOME}/.nvbench-example-venv-${py_version}-${env_name}"
    rm -rf "${NVBENCH_PYTHON_VENV}"

    echo "::group::Python examples: ${env_name}"
    setup_python_env "${py_version}"

    python --version
    nvcc --version
    export CXX="$(which g++)"
    export CUDACXX="$(which nvcc)"
    export CUDAHOSTCXX="$(which g++)"
    python -m pip install --upgrade pip

    echo "Installing wheel: ${CUDA_BENCH_WHEEL_PATH} with extra: ${cuda_extra}"
    python -m pip install "${CUDA_BENCH_WHEEL_PATH}[${cuda_extra}]"
    python -m pip install -r "${requirements_file}"

    python /workspace/ci/run_python_examples.py "${runner_args[@]}"
    echo "::endgroup::"
}

run_example_env \
    core-cccl \
    pr,example-cpu,core-cccl

run_example_env \
    numba-cupy \
    numba,cupy,cuda-compute

run_example_env \
    autotune \
    autotune

if [[ "${include_heavy_examples:-0}" == "1" ]]; then
    run_example_env \
        torch \
        torch

    run_example_env \
        cute \
        cute
fi
