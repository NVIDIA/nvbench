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
EXAMPLE_REQUIREMENTS_FILE="/workspace/python/examples/requirements.txt"
CONSTRAINTS_FILE="${EXAMPLE_REQUIREMENTS_DIR}/constraints-${cuda_extra}-py${py_version//./}.txt"
FREEZE_DIR="$(mktemp -d)"
constraints_candidate_printed=0

print_constraints_candidate() {
    if [[ "${floating_deps:-0}" != "1" || "${constraints_candidate_printed}" == "1" ]]; then
        return 0
    fi
    constraints_candidate_printed=1
    python /workspace/ci/format_python_example_constraints.py \
        --constraints-file "${CONSTRAINTS_FILE}" \
        --requirements-file "${EXAMPLE_REQUIREMENTS_FILE}" \
        --freeze-dir "${FREEZE_DIR}" \
        --python-version "${py_version}" \
        --cuda-extra "${cuda_extra}"
}

cleanup() {
    local status=$?
    print_constraints_candidate || true
    rm -rf "${FREEZE_DIR}"
    exit "${status}"
}

trap cleanup EXIT

if [[ ! -d "${WHEELHOUSE_DIR}" ]]; then
    echo "Error: Wheelhouse directory not found: ${WHEELHOUSE_DIR}" >&2
    echo "Build or download the cuda-bench wheel into wheelhouse before running example tests." >&2
    exit 1
fi

python_tag="cp${py_version//./}"
mapfile -t cuda_bench_wheels < <(
    find "${WHEELHOUSE_DIR}" -maxdepth 1 -name "cuda_bench-*-${python_tag}-*.whl" -print
)

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
    echo "Error: Missing constraints file: ${CONSTRAINTS_FILE}" >&2
    exit 1
fi

if [[ "${#cuda_bench_wheels[@]}" -eq 0 ]]; then
    echo "Error: No cuda-bench wheel for Python ${py_version} found in ${WHEELHOUSE_DIR}" >&2
    echo "Contents of ${WHEELHOUSE_DIR}:" >&2
    ls -la "${WHEELHOUSE_DIR}/" || true
    exit 1
fi
if [[ "${#cuda_bench_wheels[@]}" -gt 1 ]]; then
    echo "Error: Multiple cuda-bench wheels for Python ${py_version} found in ${WHEELHOUSE_DIR}" >&2
    printf '  %s\n' "${cuda_bench_wheels[@]}" >&2
    exit 1
fi

CUDA_BENCH_WHEEL_PATH="${cuda_bench_wheels[0]}"

run_example_env() {
    (
    set -euo pipefail

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
    CXX="$(which g++)" || { echo "Error: g++ not found in PATH" >&2; exit 1; }
    CUDACXX="$(which nvcc)" || { echo "Error: nvcc not found in PATH" >&2; exit 1; }
    CUDAHOSTCXX="$CXX"
    export CXX CUDACXX CUDAHOSTCXX
    python -m pip install --upgrade pip

    echo "Installing wheel: ${CUDA_BENCH_WHEEL_PATH} with extra: ${cuda_extra}"
    if [[ "${floating_deps:-0}" == "1" ]]; then
        python -m pip install "${CUDA_BENCH_WHEEL_PATH}[${cuda_extra}]"
        python -m pip install -r "${requirements_file}"
    else
        python -m pip install -c "${CONSTRAINTS_FILE}" "${CUDA_BENCH_WHEEL_PATH}[${cuda_extra}]"
        python -m pip install -c "${CONSTRAINTS_FILE}" -r "${requirements_file}"
    fi

    if [[ "${floating_deps:-0}" == "1" ]]; then
        local freeze_file="${FREEZE_DIR}/${env_name}.txt"
        python -m pip freeze --exclude-editable | sort -f > "${freeze_file}"
        echo "::group::Full pip freeze: ${env_name}"
        cat "${freeze_file}"
        echo "::endgroup::"
    fi

    python /workspace/ci/run_python_examples.py "${runner_args[@]}"
    echo "::endgroup::"
    )
}

run_example_env_and_record_status() {
    local env_name="$1"
    run_example_env "$@" &
    local pid=$!
    local status=0
    if wait "${pid}"; then
        status=0
    else
        status=$?
    fi
    if [[ "${status}" -ne 0 ]]; then
        echo "Error: Python example environment '${env_name}' failed" >&2
        overall_status=1
    fi
}

overall_status=0
run_example_env_and_record_status \
    core-cccl \
    pr,example-cpu,core-cccl

run_example_env_and_record_status \
    numba-cupy \
    numba,cupy,cuda-compute

run_example_env_and_record_status \
    autotune \
    autotune

if [[ "${include_heavy_examples:-0}" == "1" ]]; then
    run_example_env_and_record_status \
        torch \
        torch

    run_example_env_and_record_status \
        cute \
        cute
fi

print_constraints_candidate
exit "${overall_status}"
