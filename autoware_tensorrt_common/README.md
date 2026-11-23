# autoware_tensorrt_common

This package provides a high-level API to work with TensorRT. This library simplifies the process of loading, building, and executing TensorRT inference engines using ONNX models. It also includes utilities for profiling and managing TensorRT execution contexts, making it easier to integrate TensorRT-based packages in Autoware.

## Usage

Here is an example usage of the library. For the full API documentation, please refer to the doxygen documentation (see header [file](include/autoware/tensorrt_common/tensorrt_common.hpp)).

```c++
#include <autoware/tensorrt_common/tensorrt_common.hpp>

#include <memory>
#include <utility>
#include <vector>

using autoware::tensorrt_common::TrtCommon;
using autoware::tensorrt_common::TrtCommonConfig;
using autoware::tensorrt_common::TensorInfo;
using autoware::tensorrt_common::NetworkIO;
using autoware::tensorrt_common::ProfileDims;

std::unique_ptr<TrtCommon> trt_common_;
```

### Create a tensorrt common instance and setup engine

- With minimal configuration.

```c++
trt_common_ = std::make_unique<TrtCommon>(TrtCommonConfig("/path/to/onnx/model.onnx"));
trt_common_->setup();
```

- With full configuration.

```c++
trt_common_ = std::make_unique<TrtCommon>(TrtCommonConfig("/path/to/onnx/model.onnx", "fp16", "/path/to/engine/model.engine", (1ULL << 30U), -1, false));

std::vector<NetworkIO> network_io{
  NetworkIO("sample_input", {3, {-1, 64, 512}}), NetworkIO("sample_output", {1, {50}})};
std::vector<ProfileDims> profile_dims{
  ProfileDims("sample_input", {3, {1, 64, 512}}, {3, {3, 64, 512}}, {3, {9, 64, 512}})};

auto network_io_ptr = std::make_unique<std::vector<NetworkIO>>(network_io);
auto profile_dims_ptr = std::make_unique<std::vector<ProfileDims>>(profile_dims);

trt_common_->setup(std::move(profile_dims_ptr), std::move(network_io_ptr));
```

By defining network IO names and dimensions, an extra shapes validation will be performed after building / loading engine. This is useful to ensure the model is compatible with current code for preprocessing as it might consists of operations dependent on tensor shapes.

Profile dimension is used to specify the min, opt, and max dimensions for dynamic shapes.

Network IO or / and profile dimensions can be omitted if not needed.

### Setting input and output tensors

```c++
bool success = true;
success &= trt_common_->setTensor("sample_input", sample_input_d_.get(), nvinfer1::Dims{3, {var_size, 64, 512}});
success &= trt_common_->setTensor("sample_output", sample_output_d_.get());
return success;
```

### Execute inference

```c++
auto success = trt_common_->enqueueV3(stream_);
return success;
```

## AMD GPUs (ROCm)

When CUDA is not found, the package builds against [MIGraphX](https://github.com/ROCm/AMDMIGraphX) on ROCm instead, with the same classes and headers. The ROCm build puts `NvInfer.h`, `NvInferRuntime.h` and `NvOnnxParser.h` headers on the include path of every package that links `autoware_tensorrt_common`; they declare only the TensorRT types this API uses (`nvinfer1::Dims`, `DataType`, `TensorIOMode`, the logger, profiler and INT8 calibrator interfaces), so code written against `TrtCommon` compiles unchanged.

```bash
colcon build --packages-up-to autoware_tensorrt_common \
  --cmake-args -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx942
```

Pass `-DAUTOWARE_CUDA_USE_HIP=ON` to build with MIGraphX on a machine that also has CUDA, and to every package that links `autoware_tensorrt_common` there. The installed package sets `autoware_tensorrt_common_USE_HIP` to the backend it was built with, and stops the configuration of a package that links it with the other one. ROCm or MIGraphX installed outside `/opt/rocm` needs its prefix in `CMAKE_PREFIX_PATH`.

What differs from TensorRT:

- The compiled program is cached next to the engine path as `<engine name>.<gfx target>.<precision>.mxr`, with a `.mxr.key` file recording the MIGraphX version, GPU and precision; a program built by another MIGraphX, GPU, precision or optimization profile is rebuilt. The `.engine` file itself is not written.
- Dynamic shapes: `setup()` compiles for the `opt` dimensions of each optimization profile, and `enqueueV3()` compiles once more for every other input shape it meets (within the profile), which takes as long as the first build.
- `fp16` quantizes the model with MIGraphX. `int8` with `TrtConvCalib` calibrates from the batches of the given calibrator and runs only at the shapes `setup()` compiled; the per-layer settings of `CalibrationConfig` and TensorRT calibration caches are not used.
- `getNetwork()` and `getBuilderConfig()` return null, per-layer profiling reports nothing, and DLA (`dla_core_id`) and TensorRT's built-in plugins are not available. Libraries passed as `plugin_paths` are loaded as before, so they can register MIGraphX custom operators.
