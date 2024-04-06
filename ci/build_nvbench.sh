#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="nvbench-ci"

CMAKE_OPTIONS=""

configure_and_build_preset "NVBench" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
