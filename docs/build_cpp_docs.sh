#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/sphinx-cpp/_build"
DOXYGEN_DIR="${SCRIPT_DIR}/sphinx-cpp/_doxygen"

mkdir -p "${BUILD_DIR}" "${DOXYGEN_DIR}"

echo "Running Doxygen for C++ API..."
(cd "${SCRIPT_DIR}/sphinx-cpp" && doxygen Doxyfile)

echo "Building Sphinx C++ docs..."
sphinx-build -b html "${SCRIPT_DIR}/sphinx-cpp" "${BUILD_DIR}"

echo "C++ docs available at ${BUILD_DIR}/index.html"
