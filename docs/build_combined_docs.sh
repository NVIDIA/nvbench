#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/sphinx-combined/_build"
DOXYGEN_DIR="${SCRIPT_DIR}/sphinx-combined/_doxygen"

mkdir -p "${BUILD_DIR}" "${DOXYGEN_DIR}"

echo "Running Doxygen for combined C++ API..."
(cd "${SCRIPT_DIR}/sphinx-combined" && doxygen Doxyfile)

echo "Building combined Sphinx docs..."
sphinx-build -b html "${SCRIPT_DIR}/sphinx-combined" "${BUILD_DIR}"

echo "Combined docs available at ${BUILD_DIR}/index.html"
