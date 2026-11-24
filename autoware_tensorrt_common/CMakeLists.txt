cmake_minimum_required(VERSION 3.17)
project(autoware_tensorrt_common)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tensorrt_cmake_module REQUIRED)

option(AUTOWARE_CUDA_USE_HIP "Build with MIGraphX for AMD GPUs even if CUDA is found" OFF)

if(NOT AUTOWARE_CUDA_USE_HIP)
  find_package(CUDA)
  find_package(CUDAToolkit)
  find_package(TENSORRT)
endif()

# Without CUDA, AMD GPUs run the models with MIGraphX through the same API.
set(${PROJECT_NAME}_USE_HIP FALSE)
set(${PROJECT_NAME}_BACKEND tensorrt)
if(NOT CUDAToolkit_FOUND)
  find_package(hip CONFIG QUIET PATHS $ENV{ROCM_PATH} /opt/rocm)
  find_package(migraphx CONFIG QUIET PATHS $ENV{ROCM_PATH} /opt/rocm)
  if(hip_FOUND AND migraphx_FOUND)
    set(${PROJECT_NAME}_USE_HIP TRUE)
    set(${PROJECT_NAME}_BACKEND migraphx)
    message(STATUS "cuda is not found, building autoware_tensorrt_common with MIGraphX ${migraphx_VERSION}")
  elseif(AUTOWARE_CUDA_USE_HIP)
    message(FATAL_ERROR "AUTOWARE_CUDA_USE_HIP is ON, but HIP or MIGraphX is not found. "
      "Add the ROCm and MIGraphX prefixes to CMAKE_PREFIX_PATH or set ROCM_PATH.")
  endif()
endif()

if(NOT ${PROJECT_NAME}_USE_HIP AND NOT (CUDAToolkit_FOUND AND TENSORRT_FOUND))
  message(WARNING "cuda, tensorrt libraries are not found")
  return()
endif()

if(NOT ${PROJECT_NAME}_USE_HIP AND TENSORRT_VERSION VERSION_LESS 10.0)
  message(WARNING "Unsupported version TensorRT ${TENSORRT_VERSION} detected. This package requires TensorRT 10.0 or later.")
  return()
endif()

if(${PROJECT_NAME}_USE_HIP)
  add_library(${PROJECT_NAME} SHARED
    src/migraphx_common.cpp
    src/onnx_model_io.cpp
    src/tensorrt_common_io.cpp
    src/profiler.cpp
  )

  # MIGraphX stays private: the installed headers need only HIP.
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      hip::host
      rclcpp::rclcpp
    PRIVATE
      migraphx::c
      stdc++fs
  )

  # Also set by the stand-in NvInfer.h; defining it on the target lets tensorrt_common.hpp stop a
  # consumer whose include path finds the real TensorRT headers first.
  target_compile_definitions(${PROJECT_NAME} PUBLIC AUTOWARE_TENSORRT_COMMON_MIGRAPHX=1)

  # migraphx/include holds NvInfer.h headers with the TensorRT types used by this API.
  target_include_directories(${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/migraphx/include>
      $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
      $<INSTALL_INTERFACE:include/${PROJECT_NAME}/migraphx>
  )
else()
  add_library(${PROJECT_NAME} SHARED
    src/tensorrt_common.cpp
    src/tensorrt_common_io.cpp
    src/profiler.cpp
  )

  target_link_libraries(${PROJECT_NAME}
    CUDA::cudart
    rclcpp::rclcpp
    stdc++fs
    ${TENSORRT_LIBRARIES}
  )

  target_include_directories(${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
      ${TENSORRT_INCLUDE_DIRS}
  )

  target_compile_definitions(${PROJECT_NAME} PRIVATE
    TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
  )
endif()

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# TODO(amadeuszsz): Remove -Wno-deprecated-declarations once removing implicit quantization
target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wno-deprecated-declarations
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # These don't pass yet, disable them for now
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(autoware_cuda_utils REQUIRED)
  # Each case parses the dummy network and most build an engine from it, which TensorRT does not
  # do quickly even for a graph this small.
  ament_add_gtest(${PROJECT_NAME}_test
    test/tensorrt_common_test.cpp
    TIMEOUT 300
  )
  if(TARGET ${PROJECT_NAME}_test)
    target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
    target_include_directories(${PROJECT_NAME}_test PRIVATE
      ${autoware_cuda_utils_INCLUDE_DIRS}
    )
    target_compile_definitions(${PROJECT_NAME}_test PRIVATE
      DUMMY_ONNX_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/data/dummy_network.onnx"
    )
    if(${PROJECT_NAME}_USE_HIP)
      # The HIP build of autoware_cuda_utils puts its cuda_runtime_api.h on the target only.
      target_link_libraries(${PROJECT_NAME}_test autoware_cuda_utils::autoware_cuda_utils)
    endif()
  endif()

  if(${PROJECT_NAME}_USE_HIP)
    ament_add_gtest(${PROJECT_NAME}_migraphx_test
      test/migraphx_common_test.cpp
      TIMEOUT 600
    )
    if(TARGET ${PROJECT_NAME}_migraphx_test)
      # The test also calls MIGraphX itself, to tell which shapes it compiles and to run the
      # plugin's operator.
      target_link_libraries(${PROJECT_NAME}_migraphx_test
        ${PROJECT_NAME}
        autoware_cuda_utils::autoware_cuda_utils
        migraphx::c
      )
      # A plugin library as consumers pass in plugin_paths: it registers a MIGraphX operator.
      add_library(${PROJECT_NAME}_test_plugin MODULE test/custom_op_plugin.cpp)
      target_link_libraries(${PROJECT_NAME}_test_plugin PRIVATE migraphx::c hip::host)
      set_target_properties(${PROJECT_NAME}_test_plugin PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
      )
      add_dependencies(${PROJECT_NAME}_migraphx_test ${PROJECT_NAME}_test_plugin)
      target_compile_definitions(${PROJECT_NAME}_migraphx_test PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
        TEST_PLUGIN_PATH="$<TARGET_FILE:${PROJECT_NAME}_test_plugin>"
      )
      # conv_profiler.hpp narrows the int64_t dimensions of TensorRT 10, which consumers do not
      # see because they include it as a system header.
      target_compile_options(${PROJECT_NAME}_migraphx_test PRIVATE -Wno-narrowing)
    endif()
  endif()
endif()

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_${PROJECT_NAME})

if(${PROJECT_NAME}_USE_HIP)
  install(DIRECTORY migraphx/include/ DESTINATION include/${PROJECT_NAME}/migraphx)
  ament_export_include_directories("include/${PROJECT_NAME}/migraphx")
  # CUDA and TensorRT are not exported: a consumer that found CUDA would stop at the backend check.
  ament_export_dependencies("rclcpp")
  set(${PROJECT_NAME}_GPU_CONFIG_EXTRAS cmake/find_hip.cmake)
else()
  ament_export_dependencies(
    "CUDA"
    "CUDAToolkit"
    "rclcpp"
    "tensorrt_cmake_module"
    "TENSORRT"
  )
endif()

ament_package(CONFIG_EXTRAS ${${PROJECT_NAME}_GPU_CONFIG_EXTRAS} cmake/check_gpu_backend.cmake.in)
