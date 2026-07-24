#!/bin/bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 [-py-version <python_version>] [-cuda-version <cuda_version>] [-host <host_tag>] [-devcontainer-version <version>]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"

die_usage() {
    echo "Error: $1" >&2
    echo "$usage" >&2
    exit 1
}

require_option_value() {
    local option_name="$1"
    local option_value="${2-}"
    if [[ -z "${option_value}" || "${option_value}" == -* ]]; then
        die_usage "${option_name} requires a value"
    fi
}

if ! parse_python_args "$@"; then
    echo "$usage" >&2
    exit 1
fi

cuda_version="13.3"
host_tag="gcc15"
devcontainer_version="26.08"
while [[ $# -gt 0 ]]; do
    case $1 in
        -cuda-version=*)
            [[ -n "${1#*=}" ]] || die_usage "-cuda-version requires a value"
            cuda_version="${1#*=}"
            shift
            ;;
        -cuda-version)
            require_option_value "-cuda-version" "${2-}"
            cuda_version="$2"
            shift 2
            ;;
        -host=*)
            [[ -n "${1#*=}" ]] || die_usage "-host requires a value"
            host_tag="${1#*=}"
            shift
            ;;
        -host)
            require_option_value "-host" "${2-}"
            host_tag="$2"
            shift 2
            ;;
        -devcontainer-version=*)
            [[ -n "${1#*=}" ]] || die_usage "-devcontainer-version requires a value"
            devcontainer_version="${1#*=}"
            shift
            ;;
        -devcontainer-version)
            require_option_value "-devcontainer-version" "${2-}"
            devcontainer_version="$2"
            shift 2
            ;;
        -py-version=*)
            [[ -n "${1#*=}" ]] || die_usage "-py-version requires a value"
            shift
            ;;
        -py-version)
            require_option_value "-py-version" "${2-}"
            shift 2
            ;;
        *)
            die_usage "Unknown option: $1"
            ;;
    esac
done

if [[ -z "${py_version:-}" ]]; then
    py_version="3.14"
fi

cuda_major="${cuda_version%%.*}"
if [[ "${cuda_major}" != "12" && "${cuda_major}" != "13" ]]; then
    die_usage "Unsupported CUDA version: $cuda_version"
fi
cuda_extra="cu${cuda_major}"
readonly cuda_extra
cuda_image="rapidsai/devcontainers:${devcontainer_version}-cpp-${host_tag}-cuda${cuda_version}"
readonly cuda_image

if [[ -z "${HOST_WORKSPACE:-}" ]]; then
  HOST_WORKSPACE="$(cd "${ci_dir}/.." && pwd)"
fi
readonly HOST_WORKSPACE

echo "::group::🧪 Testing editable cuda-bench install on ${cuda_image}"
(
  set -x
  # Prevent GHA runners from exhausting available storage with leftover images,
  # even when the containerized build or tests fail.
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    trap 'docker rmi -f "${cuda_image}" || true' EXIT
  fi

  docker pull "${cuda_image}"
  docker run --rm -i \
      --workdir /workspace \
      --mount "type=bind,source=${HOST_WORKSPACE},target=/workspace/" \
      --env "py_version=${py_version}" \
      --env "cuda_extra=${cuda_extra}" \
      "${cuda_image}" \
      /bin/bash -euo pipefail <<'EOF'
  source /workspace/ci/pyenv_helper.sh
  setup_python_env "${py_version}"

  cd /workspace/python

  python --version
  nvcc --version
  export CXX="$(which g++)"
  export CUDACXX="$(which nvcc)"
  export CUDAHOSTCXX="$(which g++)"
  # Build one supported architecture only; this job validates editable
  # packaging, not GPU execution or architecture coverage.
  export CUDAARCHS="75-real"

  python -m pip install --upgrade pip
  python -m pip install -e ".[${cuda_extra},tools]" pytest

  cd /tmp

  python - <<'PY'
import importlib.util

from cuda.bench._paths import _get_cuda_major_version

cuda_major = _get_cuda_major_version()
extension_name = f"cuda.bench.cu{cuda_major}._nvbench"
extension_spec = importlib.util.find_spec(extension_name)
if extension_spec is None:
    raise RuntimeError(f"{extension_name} is not discoverable")

print(f"Found editable extension module: {extension_spec.origin}")
PY

  python -m pytest -v \
    /workspace/python/test/test_benchmark_result.py \
    /workspace/python/test/test_cuda_bench_nvbench.py \
    /workspace/python/test/test_cuda_bench_paths.py \
    /workspace/python/test/test_nvbench_compare_robust.py \
    /workspace/python/test/test_nvbench_json_summary.py \
    /workspace/python/test/test_nvbench_tooling_deps.py
EOF
)
echo "::endgroup::"
