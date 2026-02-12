#  Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 
#  SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
#  SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10...3.31)
project(nvtt_samples)

include(CheckLanguage)
enable_language(CXX)
set(CMAKE_CUDA_STANDARD 17) # Since CMake <= 3.24 doesn't know that CUDA 12 supports C++20
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  enable_language(CUDA)
endif()

#####################################################################################
# Turn off libraries we don't need for compilation speed
option(NVPRO2_ENABLE_nvaftermath OFF)
option(NVPRO2_ENABLE_nvapp OFF)
option(NVPRO2_ENABLE_nvgl OFF)
option(NVPRO2_ENABLE_nvgpu_monitor OFF)
option(NVPRO2_ENABLE_nvgui OFF)
option(NVPRO2_ENABLE_nvshaders_host OFF)
option(NVPRO2_ENABLE_nvslang OFF)
option(NVPRO2_ENABLE_nvvk OFF)
option(NVPRO2_ENABLE_nvvkglsl OFF)
option(NVPRO2_ENABLE_nvvkgltf OFF)
# look for nvpro_core or download it using the FindNvproCore2 script
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

#####################################################################################
# Find NVTT; see cmake/FindNVTT.cmake for a script you can use in your own projects
find_package(NVTT REQUIRED)

#####################################################################################
# Common files
file(GLOB _COMMON_FILES "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/common/*.h")

#####################################################################################
# Function that adds a sample with the files in a given directory, linking with NVTT.
function(add_nvtt_sample directoryName)
  # Set the project name and output directory
  set(_TARGET_NAME "nvtt_${directoryName}")
  
  # Add the executable; put common files in a nice MSVC folder
  file(GLOB _SAMPLE_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.cu"
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.h"
  )
  add_executable("${_TARGET_NAME}" "${_SAMPLE_SOURCES}" "${_COMMON_FILES}")
  source_group("common" FILES ${_COMMON_FILES})
  
  # Link with NVTT and nvpro_core2.
  target_link_libraries("${_TARGET_NAME}" PRIVATE
    nvpro2::nvimageformats
    nvpro2::nvutils
    nvtt
  )
  target_include_directories("${_TARGET_NAME}"
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/common"
  )
  add_project_definitions("${_TARGET_NAME}")
  
  # Copy NVTT to the output directory.
  copy_to_runtime_and_install(${_TARGET_NAME} AUTO)
endfunction()

#####################################################################################
# Add individual samples
add_nvtt_sample("c_wrapper_demo")
add_nvtt_sample("compress")
if(CMAKE_CUDA_COMPILER)
  add_nvtt_sample("cuda_input")
  # Copy sample images to the install directory.
  copy_to_runtime_and_install(nvtt_cuda_input
    LOCAL_DIRS ${CMAKE_CURRENT_LIST_DIR}/images
  )
else()
  message("Skipping cuda_input sample: CUDA compiler not found.")
endif()
add_nvtt_sample("ddsinfo")
add_nvtt_sample("decompress")
add_nvtt_sample("detach")
add_nvtt_sample("imgdiff")
add_nvtt_sample("mini_bc7")
add_nvtt_sample("mipmap")
add_nvtt_sample("stitch")
add_nvtt_sample("to_slope_space")
