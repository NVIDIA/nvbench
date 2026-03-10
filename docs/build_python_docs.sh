#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/sphinx-python/_build"

mkdir -p "${BUILD_DIR}"

echo "Building Sphinx Python docs..."
sphinx-build -b html "${SCRIPT_DIR}/sphinx-python" "${BUILD_DIR}"

echo "Python docs available at ${BUILD_DIR}/index.html"
