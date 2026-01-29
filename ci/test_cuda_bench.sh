#!/bin/bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> -cuda-version <cuda_version>"

source "$ci_dir/util/python/common_arg_parser.sh"

# Parse arguments including CUDA version
parse_python_args "$@"

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

# Check if py_version was provided (this script requires it)
require_py_version "$usage" || exit 1

if [[ -z "$cuda_version" ]]; then
    echo "Error: -cuda-version is required"
    echo "$usage"
    exit 1
fi

# Map cuda_version to full version and set CUDA extra
if [[ "$cuda_version" == "12" ]]; then
    cuda_full_version="12.9.1"
    cuda_extra="cu12"
elif [[ "$cuda_version" == "13" ]]; then
    cuda_full_version="13.0.1"
    cuda_extra="cu13"
else
    echo "Error: Unsupported CUDA version: $cuda_version"
    exit 1
fi

# Use the same rapidsai/ci-wheel images as the build
readonly devcontainer_version=25.12
readonly devcontainer_distro=rockylinux8

if [[ "$(uname -m)" == "aarch64" ]]; then
  readonly cuda_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda_full_version}-${devcontainer_distro}-py${py_version}-arm64
else
  readonly cuda_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda_full_version}-${devcontainer_distro}-py${py_version}
fi

echo "::group::🧪 Testing CUDA ${cuda_version} wheel on ${cuda_image}"
(
  set -x
  docker pull $cuda_image
  docker run --rm -i \
      --workdir /workspace \
      --gpus all \
      --mount type=bind,source=$(pwd),target=/workspace/ \
      --env py_version=${py_version} \
      --env cuda_version=${cuda_version} \
      --env cuda_extra="${cuda_extra}" \
      $cuda_image \
      /workspace/ci/test_cuda_bench_inner.sh
  # Prevent GHA runners from exhausting available storage with leftover images:
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    docker rmi -f $cuda_image
  fi
)
echo "::endgroup::"
