#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#  SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION
#  SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#####################################################################################
# Add sample packages
#
_add_project_definitions(nvtt_samples)
_add_package_Cuda()
_add_nvpro_core_lib()

#####################################################################################
_find_package_OpenMP()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#####################################################################################
# Find NVTT
if(NOT ("${CMAKE_SIZEOF_VOID_P}" EQUAL 8))
  message(WARNING "The NVTT samples require a 64-bit build.")
  return()
endif()

if(WIN32)
  if(NOT ("${CMAKE_VS_PLATFORM_TOOLSET}" MATCHES "v14*"))
    message(WARNING "As of this writing, the NVTT C++ API is only guaranteed to be compatible with MSVC v14x (e.g. v142, Visual Studio 2019) toolsets. You may get unexpected results running the C++ samples, but any C samples should work.")
  endif()
endif()

# Find NVTT.
if(NOT NVTT_DIR)
  set(_SEARCH_PATHS "$ENV{NVTT_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/../nvtt" "C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools")
  
  foreach(_PATH ${_SEARCH_PATHS})
    file(GLOB _DLL_POSSIBILITIES "${_PATH}/nvtt*.dll" "${_PATH}/libnvtt.so.*")
    if(_DLL_POSSIBILITIES) # If this folder contains a DLL matching the NVTT DLL pattern
      set(NVTT_DIR "${_PATH}")
      break()
    endif()
  endforeach()
  
  if(NOT NVTT_DIR)
    message(WARNING "NVTT not found! Please install NVTT from https://developer.nvidia.com/nvidia-texture-tools-exporter and set the CMake NVTT_DIR variable to the folder containing nvtt*.dll (e.g. C:\\Program Files\\NVIDIA Corporation\\NVIDIA Texture Tools).")
    return()
  endif()
endif()

# Get the NVTT shared library name.
file(GLOB _NVTT_SL_POSSIBILITIES "${NVTT_DIR}/nvtt*.dll" "${NVTT_DIR}/libnvtt.so.*")
if(NOT _NVTT_SL_POSSIBILITIES)
  message(WARNING "NVTT_DIR didn't contain an NVTT shared library of the form nvtt*.dll or libnvtt.so.*! Is NVTT_DIR set correctly? NVTT_DIR was ${NVTT_DIR}")
  return()
else()
  list(LENGTH _NVTT_SL_POSSIBILITIES _NVTT_SL_POSSIBILITIES_LEN)
  math(EXPR _NVTT_SL_IDX ${_NVTT_SL_POSSIBILITIES_LEN}-1)
  list(GET _NVTT_SL_POSSIBILITIES ${_NVTT_SL_IDX} _NVTT_SL)
endif()

# Find the NVTT linker library on Windows.
if(WIN32)
  if(NOT NVTT_LIB)
    file(GLOB _NVTT_LIB_ALL "${NVTT_DIR}/lib/x64-v*/nvtt*.lib")
    if(NOT _NVTT_LIB_ALL)
      message(WARNING "Found nvtt.dll in ${NVTT_DIR}, but was unable to find nvtt.lib in ${NVTT_DIR}/lib/... ! Please check the NVTT directory and this CMake script to see if the path is correct.")
      return()
    endif()
    list(LENGTH _NVTT_LIB_ALL _NVTT_LIB_LEN)
    math(EXPR _NVTT_LIB_IDX ${_NVTT_LIB_LEN}-1)
    list(GET _NVTT_LIB_ALL ${_NVTT_LIB_IDX} NVTT_LIB)
  endif()
endif()

# Print out information to help with error reports.
message(STATUS "NVTT Shared Library: ${_NVTT_SL}")
message(STATUS "NVTT .lib (Windows): ${NVTT_LIB}")

file(GLOB _COMMON_FILES "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/common/*.h")

# Build with an rpath so that on Linux, libraries are searched for in the local directory.
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "$\{ORIGIN\}")

#####################################################################################
# Function that adds a sample with the files in a given directory, linking with NVTT.
function(add_nvtt_sample directoryName)
  # Set the project name and output directory
  set(_TARGET_NAME "nvtt_${directoryName}")
  
  # Add the executable; put common files in a nice MSVC folder
  file(GLOB _SAMPLE_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/${directoryName}/*.h"
  )
  add_executable("${_TARGET_NAME}" "${_SAMPLE_SOURCES}" "${_COMMON_FILES}")
  source_group("common" FILES ${_COMMON_FILES})
  
  # Link with NVTT
  if(WIN32)
    target_link_libraries("${_TARGET_NAME}" PRIVATE "${NVTT_LIB}")
  else()
    target_link_libraries("${_TARGET_NAME}" PRIVATE "${_NVTT_SL}")
  endif()
  target_link_libraries("${_TARGET_NAME}" PRIVATE ${PLATFORM_LIBRARIES} nvpro_core)
  target_include_directories("${_TARGET_NAME}"
    PRIVATE
    "${NVTT_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/common"
  )
  
  # Also link with the libraries we added (CUDA) if we have them.
  foreach(DEBUGLIB ${LIBRARIES_DEBUG})
    target_link_libraries(${_TARGET_NAME} PRIVATE debug ${DEBUGLIB})
  endforeach(DEBUGLIB)

  foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
    target_link_libraries(${_TARGET_NAME} PRIVATE optimized ${RELEASELIB})
  endforeach(RELEASELIB)
  
  # Copy NVTT to the output directory.
  add_custom_command(
    TARGET "${_TARGET_NAME}"
    POST_BUILD
    COMMAND cmake -E copy_if_different "${_NVTT_SL}" "$<TARGET_FILE_DIR:${_TARGET_NAME}>")
  _finalize_target(${_TARGET_NAME})
  
  # Remove global definitions from _add_project_definitions for next subproject.
  remove_definitions(-DPROJECT_NAME -DPROJECT_RELDIRECTORY -DPROJECT_DOWNLOAD_RELDIRECTORY)
endfunction()

#####################################################################################
# Add individual samples
add_nvtt_sample("c_wrapper_demo")
add_nvtt_sample("compress")
add_nvtt_sample("cuda_input")
add_nvtt_sample("decompress")
add_nvtt_sample("imgdiff")
add_nvtt_sample("mini_bc7")
add_nvtt_sample("mipmap")
add_nvtt_sample("to_slope_space")

#####################################################################################
# Copy sample images to the install directory.
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cuda_input.raw CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cuda_input.raw CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
# Copy the NVTT shared libraries to the install directory.
install(FILES "${_NVTT_SL}" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
install(FILES "${_NVTT_SL}" CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
