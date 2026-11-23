# Copyright 2026 The Autoware Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The ROCm counterpart of find_cccl.cmake. ROCm installed outside /opt/rocm
# needs its prefix in CMAKE_PREFIX_PATH or ROCM_PATH.

find_package(hip CONFIG REQUIRED PATHS $ENV{ROCM_PATH} /opt/rocm)
find_package(rocthrust CONFIG REQUIRED PATHS $ENV{ROCM_PATH} /opt/rocm)

# roc::rocthrust links hip::device, whose --offload-arch flags break host
# compilers, so consumers get only the include directories and hip::host.
if(NOT TARGET autoware_cuda_utils::rocthrust_system)
  add_library(autoware_cuda_utils::rocthrust_system INTERFACE IMPORTED GLOBAL)
  get_target_property(_autoware_cuda_utils_rocthrust_dirs roc::rocthrust
    INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(_autoware_cuda_utils_rocprim_dirs roc::rocprim
    INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(autoware_cuda_utils::rocthrust_system PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "${_autoware_cuda_utils_rocthrust_dirs};${_autoware_cuda_utils_rocprim_dirs}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
      "${_autoware_cuda_utils_rocthrust_dirs};${_autoware_cuda_utils_rocprim_dirs}"
    INTERFACE_LINK_LIBRARIES hip::host)
  unset(_autoware_cuda_utils_rocthrust_dirs)
  unset(_autoware_cuda_utils_rocprim_dirs)
endif()
