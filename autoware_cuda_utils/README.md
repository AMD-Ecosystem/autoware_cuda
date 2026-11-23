# autoware_cuda_utils

## Purpose

This package contains a library of common functions related to CUDA.

## AMD GPUs (ROCm)

When CUDA is not found, the package builds with HIP on ROCm instead, using rocThrust for Thrust. The headers are unchanged: the ROCm build puts `cuda_runtime_api.h` and `cuda_runtime.h` headers that map the CUDA runtime names used here to HIP on the include path of every package that links `autoware_cuda_utils`, and `.cu` sources are compiled as HIP.

```bash
colcon build --packages-up-to autoware_cuda_utils \
  --cmake-args -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx942
```

Pass `-DAUTOWARE_CUDA_USE_HIP=ON` to build with HIP on a machine that also has CUDA, and to every package that links `autoware_cuda_utils` there. The installed package sets `autoware_cuda_utils_USE_HIP` to the backend it was built with, and stops the configuration of a package that links it with the other one. ROCm installed outside `/opt/rocm` needs its prefix in `CMAKE_PREFIX_PATH` or `ROCM_PATH`. `CMAKE_HIP_ARCHITECTURES` is the GPU target (`gfx942` for MI300, `gfx1100` for Radeon RX 7900); without it CMake builds for the GPUs it detects.
