#!/bin/bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [-include-heavy-examples] [-floating-deps]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"

parse_python_args "$@"

include_heavy_examples=0
floating_deps=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -include-heavy-examples)
            include_heavy_examples=1
            shift
            ;;
        -floating-deps)
            floating_deps=1
            shift
            ;;
        -py-version=*)
            shift
            ;;
        -py-version)
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "$usage" >&2
            exit 1
            ;;
    esac
done

require_py_version "$usage" || exit 1

readonly cuda_extra=cu13
readonly cuda_runtime_version=13.3
readonly devcontainer_version=26.08
readonly host_tag=gcc15

readonly cuda_image=rapidsai/devcontainers:${devcontainer_version}-cpp-${host_tag}-cuda${cuda_runtime_version}

echo "::group::Testing Python examples on ${cuda_image}"
(
  set -x
  # Prevent GHA runners from exhausting available storage with leftover images,
  # even when the containerized example tests fail.
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    trap 'docker rmi -f "${cuda_image}" || true' EXIT
  fi

  docker pull "${cuda_image}"
  docker run --rm -i \
      --workdir /workspace \
      --gpus all \
      --mount "type=bind,source=$(pwd),target=/workspace/" \
      --env "py_version=${py_version}" \
      --env "cuda_extra=${cuda_extra}" \
      --env "include_heavy_examples=${include_heavy_examples}" \
      --env "floating_deps=${floating_deps}" \
      --env "PYTHONPYCACHEPREFIX=/tmp/nvbench-pycache" \
      "${cuda_image}" \
      /workspace/ci/test_python_examples_inner.sh
)
echo "::endgroup::"
