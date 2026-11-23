cmake_minimum_required(VERSION 3.22)
project(autoware_cuda_utils)

find_package(autoware_cmake REQUIRED)
autoware_package()

option(AUTOWARE_CUDA_USE_HIP "Build with HIP for AMD GPUs even if CUDA is found" OFF)

if(NOT AUTOWARE_CUDA_USE_HIP)
  find_package(CUDA)
endif()

if(NOT CUDA_FOUND)
  find_package(hip CONFIG QUIET PATHS $ENV{ROCM_PATH} /opt/rocm)
  find_package(rocthrust CONFIG QUIET PATHS $ENV{ROCM_PATH} /opt/rocm)
  if(NOT hip_FOUND OR NOT rocthrust_FOUND)
    if(AUTOWARE_CUDA_USE_HIP)
      message(FATAL_ERROR "AUTOWARE_CUDA_USE_HIP is ON, but HIP or rocThrust is not found. "
        "Add the ROCm prefix to CMAKE_PREFIX_PATH or set ROCM_PATH.")
    endif()
    message(WARNING "cuda is not found, so the autoware_cuda_utils package won't be built.")
    return()
  endif()
  message(STATUS "cuda is not found, building autoware_cuda_utils with HIP ${hip_VERSION}")
endif()

# The installed headers use Thrust and cuda_runtime_api.h, so consumers need both include paths.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(CUDA_FOUND)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/find_cccl.cmake)
  target_link_libraries(${PROJECT_NAME} INTERFACE CCCL::CCCL autoware_cuda_utils::cccl_system CUDA::cudart)
  set(${PROJECT_NAME}_GPU_CONFIG_EXTRAS cmake/find_cccl.cmake)
  set(${PROJECT_NAME}_USE_HIP FALSE)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/find_rocthrust.cmake)
  # hip/include holds cuda_runtime_api.h and cuda_runtime.h headers that map to HIP.
  target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hip/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/hip>
  )
  target_link_libraries(${PROJECT_NAME} INTERFACE autoware_cuda_utils::rocthrust_system)
  install(DIRECTORY hip/include/ DESTINATION include/${PROJECT_NAME}/hip)
  set(${PROJECT_NAME}_GPU_CONFIG_EXTRAS cmake/find_rocthrust.cmake)
  set(${PROJECT_NAME}_USE_HIP TRUE)
endif()

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_find_gtest()

  set(test_cuda_sources
    test/test_cuda_check_error.cu
    test/test_cuda_unique_ptr.cu
    test/test_cuda_utils.cu
    test/test_stream_unique_ptr.cu
    test/test_thrust_utils.cu
  )
  if(CUDA_FOUND)
    cuda_add_executable(test_autoware_cuda_utils
      ${test_cuda_sources}
      test/test_main.cpp
    )
  else()
    enable_language(HIP)
    set_source_files_properties(${test_cuda_sources} PROPERTIES LANGUAGE HIP)
    add_executable(test_autoware_cuda_utils
      ${test_cuda_sources}
      test/test_main.cpp
    )
    # HIP declares hipError_t [[nodiscard]]; some test calls deliberately ignore it.
    target_compile_options(test_autoware_cuda_utils PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value -Wno-unused-result>)
  endif()

  target_link_libraries(test_autoware_cuda_utils
    ${PROJECT_NAME}
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
  )

  target_include_directories(test_autoware_cuda_utils
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    SYSTEM
    ${GTEST_INCLUDE_DIRS}
  )

  ament_add_gtest_test(test_autoware_cuda_utils)
endif()

ament_auto_package(CONFIG_EXTRAS ${${PROJECT_NAME}_GPU_CONFIG_EXTRAS} cmake/check_gpu_backend.cmake.in)
