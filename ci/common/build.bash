#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

################################################################################
# NVBench build script for gpuCI
################################################################################

set -e

# append variable value
# Appends ${value} to ${variable}, adding a space before ${value} if
# ${variable} is not empty.
function append {
  tmp="${!1:+${!1} }${2}"
  eval "${1}=\${tmp}"
}

# log args...
# Prints out ${args[*]} with a gpuCI log prefix and a newline before and after.
function log() {
  printf "\n>>>> %s\n\n" "${*}"
}

# print_with_trailing_blank_line args...
# Prints ${args[*]} with one blank line following, preserving newlines within
# ${args[*]} but stripping any preceding ${args[*]}.
function print_with_trailing_blank_line {
  printf "%s\n\n" "${*}"
}

# echo_and_run name args...
# Echo ${args[@]}, then execute ${args[@]}
function echo_and_run {
  echo "${1}: ${@:2}"
  ${@:2}
}

# echo_and_run_timed name args...
# Echo ${args[@]}, then execute ${args[@]} and report how long it took,
# including ${name} in the output of the time.
function echo_and_run_timed {
  echo "${@:2}"
  TIMEFORMAT=$'\n'"${1} Time: %lR"
  time ${@:2}
}

# join_delimit <delimiter> [value [value [...]]]
# Combine all values into a single string, separating each by a single character
# delimiter. Eg:
# foo=(bar baz kramble)
# joined_foo=$(join_delimit "|" "${foo[@]}")
# echo joined_foo # "bar|baz|kramble"
function join_delimit {
  local IFS="${1}"
  shift
  echo "${*}"
}

################################################################################
# VARIABLES - Set up bash and environmental variables.
################################################################################

# Get the variables the Docker container set up for us: ${CXX}, ${CUDACXX}, etc.
source /etc/cccl.bashrc

# Set path.
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

# Switch to the build directory.
cd ${WORKSPACE}
mkdir -p build
cd build

# Remove any old .ninja_log file so the PrintNinjaBuildTimes step is accurate:
rm -f .ninja_log

if [[ -z "${CMAKE_BUILD_TYPE}" ]]; then
  CMAKE_BUILD_TYPE="Release"
fi

CMAKE_BUILD_FLAGS="--"

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
append CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER='${CUDACXX}'"

if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  echo "nvc++ not supported."
  exit 1
else
  if [[ "${CXX_TYPE}" == "icc" ]]; then
    echo "icc not supported."
    exit 1
  fi
  # We're using NVCC so we need to set the host compiler.
  append CMAKE_FLAGS "-DCMAKE_CXX_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-DCMAKE_CUDA_HOST_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-G Ninja"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k0"
fi

if [[ -n "${PARALLEL_LEVEL}" ]]; then
  DETERMINE_PARALLELISM_FLAGS="-j ${PARALLEL_LEVEL}"
fi

WSL=0
if [[ $(grep -i microsoft /proc/version) ]]; then
  echo "Windows Subsystem for Linux detected."
  WSL=1
fi
export WSL

#append CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=all"

append CMAKE_FLAGS "-DNVBench_ENABLE_EXAMPLES=ON"
append CMAKE_FLAGS "-DNVBench_ENABLE_TESTING=ON"
append CMAKE_FLAGS "-DNVBench_ENABLE_CUPTI=ON"
append CMAKE_FLAGS "-DNVBench_ENABLE_WERROR=ON"

# These consume a lot of time and don't currently have
# any value as regression tests.
append CMAKE_FLAGS "-DNVBench_ENABLE_DEVICE_TESTING=OFF"

# NVML doesn't work under WSL
if [[ ${WSL} -eq 0 ]]; then
  append CMAKE_FLAGS "-DNVBench_ENABLE_NVML=ON"
else
  append CMAKE_FLAGS "-DNVBench_ENABLE_NVML=OFF"
fi

if [[ -n "${@}" ]]; then
  append CMAKE_BUILD_FLAGS "${@}"
fi

append CTEST_FLAGS "--output-on-failure"

# Export variables so they'll show up in the logs when we report the environment.
export CMAKE_FLAGS
export CMAKE_BUILD_FLAGS
export CTEST_FLAGS

################################################################################
# ENVIRONMENT - Configure and print out information about the environment.
################################################################################

log "Determine system topology..."

# Set `${PARALLEL_LEVEL}` if it is unset; otherwise, this just reports the
# system topology.
source ${WORKSPACE}/ci/common/determine_build_parallelism.bash ${DETERMINE_PARALLELISM_FLAGS}

log "Get environment..."

env | sort

log "Check versions..."

# We use sed and echo below to ensure there is always one and only trailing
# line following the output from each tool.

${CXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

${CUDACXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

cmake --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

if [[ "${BUILD_TYPE}" == "gpu" ]]; then
  nvidia-smi 2>&1 | sed -Ez '$ s/\n*$/\n/'
fi

################################################################################
# BUILD
################################################################################

log "Configure..."

echo_and_run_timed "Configure" cmake .. --log-level=VERBOSE ${CMAKE_FLAGS}
configure_status=$?

log "Build..."

# ${PARALLEL_LEVEL} needs to be passed after we run
# determine_build_parallelism.bash, so it can't be part of ${CMAKE_BUILD_FLAGS}.
set +e # Don't stop on build failures.
echo_and_run_timed "Build" cmake --build . ${CMAKE_BUILD_FLAGS} -j ${PARALLEL_LEVEL}
build_status=$?
set -e

################################################################################
# TEST - Run examples and tests.
################################################################################

log "Test..."

(
  # Make sure test_status captures ctest, not tee:
  # https://stackoverflow.com/a/999259/11130318
  set -o pipefail
  echo_and_run_timed "Test" ctest ${CTEST_FLAGS} -j ${PARALLEL_LEVEL} | tee ctest_log
)

test_status=$?

################################################################################
# SUMMARY - Print status of each step and exit with failure if needed.
################################################################################

log "Summary:"
echo "- Configure Error Code: ${configure_status}"
echo "- Build Error Code: ${build_status}"
echo "- Test Error Code: ${test_status}"

if [[ "${configure_status}" != "0" ]] || \
   [[ "${build_status}" != "0" ]] || \
   [[ "${test_status}" != "0" ]]; then
     exit 1
fi
