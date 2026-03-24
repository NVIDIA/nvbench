/*
 *  Copyright 2026 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License. You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#if __has_include(<cupti_profiler_host.h>)
#include <nvbench/cupti_profiler_host.cuh>
#else
#include <nvbench/cupti_profiler_nvpw.cuh>
#endif
