#!/bin/bash

setup_python_env() {
    local py_version=$1
    local script_dir=""
    local uv_installer=""
    local venv_dir=""
    local actual_py_version=""

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        uv_installer="$(mktemp)"
        "${script_dir}/util/retry.sh" 5 30 curl -LsSf https://astral.sh/uv/install.sh -o "${uv_installer}"
        sh "${uv_installer}"
        rm -f "${uv_installer}"
        export PATH="$HOME/.local/bin:$PATH"
    fi

    venv_dir="${NVBENCH_PYTHON_VENV:-${HOME}/.nvbench-venv-${py_version}}"
    uv venv --seed --python "${py_version}" "${venv_dir}"

    if [[ -f "${venv_dir}/Scripts/activate" ]]; then
        # Windows venvs use Scripts/.
        # shellcheck source=/dev/null
        source "${venv_dir}/Scripts/activate"
    else
        # shellcheck source=/dev/null
        source "${venv_dir}/bin/activate"
    fi

    actual_py_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${actual_py_version}" != "${py_version}" ]]; then
        echo "Error: expected Python ${py_version}, got ${actual_py_version}"
        exit 1
    fi
}
