// Copyright 2026 The Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TrtCommon on AMD GPUs: the ONNX model is compiled and run with MIGraphX. A compiled program
// takes the place of the TensorRT engine and is cached on disk next to the engine path.

#include "autoware/tensorrt_common/logger.hpp"
#include "autoware/tensorrt_common/tensorrt_common.hpp"
#include "autoware/tensorrt_common/utils.hpp"
#include "onnx_model_io.hpp"

#include <migraphx/migraphx.hpp>

#include <NvInfer.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <migraphx/version.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autoware
{
namespace tensorrt_common
{
namespace
{
using Lens = std::vector<std::size_t>;

constexpr const char * kOutputParameterTag = "#output_";

std::optional<nvinfer1::DataType> toDataType(const int32_t onnx_elem_type)
{
  // ONNX TensorProto.DataType values.
  switch (onnx_elem_type) {
    case 1:
      return nvinfer1::DataType::kFLOAT;
    case 2:
      return nvinfer1::DataType::kUINT8;
    case 3:
      return nvinfer1::DataType::kINT8;
    case 6:
      return nvinfer1::DataType::kINT32;
    case 7:
      return nvinfer1::DataType::kINT64;
    case 9:
      return nvinfer1::DataType::kBOOL;
    case 10:
      return nvinfer1::DataType::kHALF;
    case 16:
      return nvinfer1::DataType::kBF16;
    case 17:
      return nvinfer1::DataType::kFP8;
    default:
      return std::nullopt;
  }
}

nvinfer1::Dims toDims(const std::vector<int64_t> & values)
{
  nvinfer1::Dims dims{};
  dims.nbDims =
    static_cast<int32_t>(std::min<std::size_t>(values.size(), nvinfer1::Dims::MAX_DIMS));
  std::copy_n(values.begin(), dims.nbDims, dims.d);
  return dims;
}

nvinfer1::Dims toDims(const Lens & lens)
{
  return toDims(std::vector<int64_t>(lens.begin(), lens.end()));
}

std::string lensRepr(const Lens & lens)
{
  return TensorInfo::dimsRepr(toDims(lens));
}

std::string migraphxVersion()
{
  return std::to_string(MIGRAPHX_VERSION_MAJOR) + "." + std::to_string(MIGRAPHX_VERSION_MINOR) +
         "." + std::to_string(MIGRAPHX_VERSION_PATCH) + "." + MIGRAPHX_VERSION_TWEAK;
}

/// The current device's architecture with its target features, e.g. "gfx942:sramecc+:xnack-".
std::optional<std::string> deviceArch()
{
  int device = 0;
  hipDeviceProp_t properties{};
  if (
    hipGetDevice(&device) != hipSuccess ||
    hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return std::nullopt;
  }
  return std::string(properties.gcnArchName);
}

/// The key a cached program must match to be loaded instead of rebuilt.
std::vector<std::string> cacheKey(const std::string & arch, const std::string & precision)
{
  return {"migraphx " + migraphxVersion(), "target " + arch, "precision " + precision};
}

constexpr const char * kProfilePrefix = "profile ";

/// Where the compiled program is cached: next to the engine path, named for the GPU and the
/// precision, so that it never replaces a TensorRT engine file.
fs::path migraphxProgramPath(const TrtCommonConfig & config, const std::string & arch)
{
  const auto gfx = arch.substr(0, arch.find(':'));
  auto path = config.engine_path;
  return path.replace_filename(path.stem().string() + "." + gfx + "." + config.precision + ".mxr");
}
}  // namespace

/**
 * @brief A compiled MIGraphX program and how its parameters bind to the model's IO tensors.
 */
struct CompiledProgram
{
  migraphx::program program;
  //!< Input shapes the program was compiled for, in model input order.
  std::vector<Lens> input_lens;
  //!< Parameter name, IO tensor index, and parameter shape.
  struct Binding
  {
    std::string name;
    int32_t index;
    migraphx::shape shape;
  };
  std::vector<Binding> bindings;
};

struct TrtCommon::MigraphxState
{
  //!< Inputs and outputs the ONNX model declares, in its order: the IO tensor indices.
  OnnxModelIO model;
  //!< Device architecture the programs are compiled for.
  std::string arch;
  //!< Program compiled by `setup()`, for the optimal profile shapes.
  std::shared_ptr<CompiledProgram> engine;
  //!< Every compiled program by input shapes, `engine` included.
  std::map<std::vector<Lens>, std::shared_ptr<CompiledProgram>> programs;
  //!< Profile the cached `engine` was built for, as `ProfileDims::toString()` lines.
  std::set<std::string> engine_profile;
  //!< Tensor addresses by IO tensor index.
  std::vector<void *> addresses;
  //!< Input shapes set by `setInputShape()`, by input index.
  std::vector<std::optional<Lens>> input_shapes;

  [[nodiscard]] int32_t numInputs() const { return static_cast<int32_t>(model.inputs.size()); }
  [[nodiscard]] int32_t numTensors() const
  {
    return static_cast<int32_t>(model.inputs.size() + model.outputs.size());
  }
  [[nodiscard]] const OnnxTensor & tensor(const int32_t index) const
  {
    return index < numInputs() ? model.inputs.at(index) : model.outputs.at(index - numInputs());
  }
  [[nodiscard]] std::optional<int32_t> indexOf(const char * name) const
  {
    if (name == nullptr) {
      return std::nullopt;
    }
    for (int32_t index = 0; index < numTensors(); ++index) {
      if (tensor(index).name == name) {
        return index;
      }
    }
    return std::nullopt;
  }
};

TrtCommon::TrtCommon(
  const TrtCommonConfig & trt_config, const std::shared_ptr<Profiler> & profiler,
  const std::vector<std::string> & plugin_paths)
: trt_config_(std::make_shared<TrtCommonConfig>(trt_config)),
  host_profiler_(profiler),
  model_profiler_(profiler)
{
  logger_ = std::make_shared<Logger>();

  // Set HIP device flags
  // note: Device flags are process-wide
  (void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync);

  // A plugin library registers its MIGraphX custom operators when it is loaded.
  for (const auto & plugin_path : plugin_paths) {
    int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
    flags |= RTLD_NODELETE;
#endif  // ENABLE_ASAN
    void * handle = dlopen(plugin_path.c_str(), flags);
    if (!handle) {
      logger_->log(
        nvinfer1::ILogger::Severity::kERROR, "Could not load plugin library %s. error %s",
        plugin_path.c_str(), dlerror());
    } else {
      logger_->log(
        nvinfer1::ILogger::Severity::kINFO, "Loaded plugin library: %s", plugin_path.c_str());
    }
  }

  if (!initialize()) {
    throw std::runtime_error("Failed to initialize MIGraphX");
  }
}

TrtCommon::~TrtCommon() = default;

bool TrtCommon::initialize()
{
  if (!fs::exists(trt_config_->onnx_path) || trt_config_->onnx_path.extension() != ".onnx") {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Invalid ONNX file path or extension");
    return false;
  }
  if (trt_config_->dla_core_id != -1) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "DLA core %d requested, but MIGraphX has no DLA",
      trt_config_->dla_core_id);
    return false;
  }

  auto arch = deviceArch();
  if (!arch) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Fail to query the GPU");
    return false;
  }

  auto model = readOnnxModelIO(trt_config_->onnx_path.string());
  if (!model) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "Fail to read the inputs and outputs of %s",
      trt_config_->onnx_path.c_str());
    return false;
  }

  // Parse once so that a model MIGraphX cannot read fails here, as it does with TensorRT.
  try {
    migraphx::parse_onnx(trt_config_->onnx_path.c_str());
  } catch (const std::exception & e) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "Fail to parse %s: %s", trt_config_->onnx_path.c_str(),
      e.what());
    return false;
  }

  if (isStronglyTyped()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kINFO,
      "Building a strongly typed network; tensor precisions are taken from the model itself");
  }
  if (trt_config_->profile_per_layer) {
    logger_->log(
      nvinfer1::ILogger::Severity::kWARNING,
      "MIGraphX reports no per-layer timing; only the host profiler is filled");
  }

  migraphx_ = std::make_unique<MigraphxState>();
  migraphx_->model = std::move(*model);
  migraphx_->arch = *arch;
  migraphx_->addresses.assign(migraphx_->numTensors(), nullptr);
  migraphx_->input_shapes.assign(migraphx_->numInputs(), std::nullopt);
  return true;
}

namespace
{
/// Compile-side helpers that need the model and configuration but not the TrtCommon object.
struct ProgramBuilder
{
  const TrtCommonConfig & config;
  const OnnxModelIO & model;
  nvinfer1::IInt8Calibrator * calibrator;
  Logger & logger;

  /// Parse the model with `input_lens` as the input shapes, quantize and compile it.
  [[nodiscard]] migraphx::program compile(const std::vector<Lens> & input_lens) const
  {
    migraphx::onnx_options options;
    for (std::size_t i = 0; i < model.inputs.size(); ++i) {
      options.set_input_parameter_shape(model.inputs[i].name, input_lens[i]);
    }
    auto program = migraphx::parse_onnx(config.onnx_path.c_str(), options);
    const migraphx::target gpu("gpu");

    if (config.precision == "fp16") {
      migraphx::quantize_fp16(program);
    } else if (config.precision == "int8") {
      quantizeInt8(program, gpu);
    }

    migraphx::compile_options compile_options;
    // Inputs and outputs live in the caller's device buffers.
    compile_options.set_offload_copy(false);
    program.compile(gpu, compile_options);
    return program;
  }

  void quantizeInt8(migraphx::program & program, const migraphx::target & gpu) const
  {
    if (!calibrator) {
      logger.log(
        nvinfer1::ILogger::Severity::kWARNING,
        "int8 requested without a calibrator: compiling the model's own precisions, which keeps "
        "any explicit quantization in it");
      return;
    }

    // The calibrator fills device buffers shaped like the inputs being compiled; MIGraphX reads
    // its calibration data from host memory.
    const auto parameter_shapes = program.get_parameter_shapes();
    const auto num_inputs = model.inputs.size();
    std::vector<void *> bindings(num_inputs, nullptr);
    std::vector<const char *> names(num_inputs);
    for (std::size_t i = 0; i < num_inputs; ++i) {
      names[i] = model.inputs[i].name.c_str();
    }

    migraphx::quantize_int8_options options;
    std::vector<std::vector<char>> host_batches;
    int32_t num_batches = 0;
    while (calibrator->getBatch(bindings.data(), names.data(), static_cast<int32_t>(num_inputs))) {
      migraphx::program_parameters batch;
      for (std::size_t i = 0; i < num_inputs; ++i) {
        const auto shape = parameter_shapes[names[i]];
        auto & host = host_batches.emplace_back(shape.bytes());
        if (
          bindings[i] == nullptr ||
          hipMemcpy(host.data(), bindings[i], host.size(), hipMemcpyDeviceToHost) != hipSuccess) {
          throw std::runtime_error(
            std::string("Fail to read calibration batch for input ") + names[i]);
        }
        batch.add(names[i], migraphx::argument(shape, host.data()));
      }
      options.add_calibration_data(batch);
      ++num_batches;
    }
    if (num_batches == 0) {
      logger.log(
        nvinfer1::ILogger::Severity::kWARNING,
        "The calibrator returned no batches, so the model is compiled without int8 quantization. "
        "MIGraphX cannot use a TensorRT calibration cache.");
      return;
    }
    logger.log(
      nvinfer1::ILogger::Severity::kINFO, "Quantizing to int8 with %d calibration batches",
      num_batches);
    migraphx::quantize_int8(program, gpu, options);
  }
};

/// Map each parameter of a compiled program to the IO tensor it binds.
std::shared_ptr<CompiledProgram> describeProgram(
  migraphx::program program, const OnnxModelIO & model)
{
  auto compiled = std::make_shared<CompiledProgram>();
  const auto num_inputs = model.inputs.size();
  compiled->input_lens.resize(num_inputs);
  const auto parameter_shapes = program.get_parameter_shapes();
  for (const char * name : parameter_shapes.names()) {
    const std::string parameter(name);
    const auto shape = parameter_shapes[name];
    int32_t index = -1;
    if (const auto tag = parameter.find(kOutputParameterTag); tag != std::string::npos) {
      const auto output = std::stoul(parameter.substr(tag + std::strlen(kOutputParameterTag)));
      if (output < model.outputs.size()) {
        index = static_cast<int32_t>(num_inputs + output);
      }
    } else {
      for (std::size_t i = 0; i < num_inputs; ++i) {
        if (model.inputs[i].name == parameter) {
          index = static_cast<int32_t>(i);
          compiled->input_lens[i] = shape.lengths();
        }
      }
    }
    if (index < 0) {
      throw std::runtime_error("Compiled program has an unexpected parameter: " + parameter);
    }
    compiled->bindings.push_back({parameter, index, shape});
  }
  compiled->program = std::move(program);
  return compiled;
}
}  // namespace

bool TrtCommon::setup(ProfileDimsPtr profile_dims, NetworkIOPtr network_io)
{
  if (!acceptOfferedIO(std::move(profile_dims), std::move(network_io))) {
    return false;
  }

  if (network_io_ && isStronglyTyped()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kINFO,
      "Strongly typed network: NetworkIO dtype overrides are not applied");
  } else if (network_io_) {
    for (const auto & io : *network_io_) {
      if (io.dtype && getTensorDataType(io.tensor_name.c_str()) != io.dtype) {
        logger_->log(
          nvinfer1::ILogger::Severity::kWARNING,
          "MIGraphX cannot change the dtype of tensor '%s' to %s; it keeps the model's",
          io.tensor_name.c_str(), dtype_name(*io.dtype));
      }
    }
  }

  auto build_engine_with_log = [this]() -> bool {
    logger_->log(nvinfer1::ILogger::Severity::kINFO, "Starting to build engine");
    auto log_thread = logger_->log_throttle(
      nvinfer1::ILogger::Severity::kINFO,
      "Applying optimizations and compiling the MIGraphX program. Please wait for a few "
      "minutes...",
      5);
    auto success = buildEngineFromOnnx();
    logger_->stop_throttle(log_thread);
    logger_->log(nvinfer1::ILogger::Severity::kINFO, "Engine build completed");
    return success;
  };

  // Load the cached program if it exists
  const auto program_path = migraphxProgramPath(*trt_config_, migraphx_->arch);
  if (fs::exists(program_path)) {
    logger_->log(nvinfer1::ILogger::Severity::kINFO, "Loading engine");
    if (!validateEngine()) {
      logger_->log(
        nvinfer1::ILogger::Severity::kWARNING,
        "Validation failed for the existing engine file. Rebuilding");
      if (!build_engine_with_log()) {
        return false;
      }
    } else if (!loadEngine()) {
      return false;
    }
    logger_->log(nvinfer1::ILogger::Severity::kINFO, "Network validation");
    if (!validateNetworkIO() || !validateProfileDims()) {
      logger_->log(
        nvinfer1::ILogger::Severity::kWARNING,
        "Network validation failed for loaded engine from file. Rebuilding engine");
      if (!build_engine_with_log()) {
        return false;
      }
    }
  } else {
    if (!build_engine_with_log()) {
      return false;
    }
  }

  if (!validateNetworkIO() || !validateProfileDims()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR,
      "Final network validation failed. Possibly the input / output of the currently "
      "deployed model has changed. Check your configuration file with the current model.");
    return false;
  }

  logger_->log(nvinfer1::ILogger::Severity::kINFO, "Engine setup completed");
  return true;
}

std::string TrtCommon::getPrecision() const
{
  return trt_config_->precision;
}

const char * TrtCommon::getIOTensorName(const int32_t index) const
{
  if (!migraphx_) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Network is not initialized");
    return nullptr;
  }
  if (index < 0 || index >= migraphx_->numTensors()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR,
      "Invalid index for I/O tensor: %d. Total I/O tensors: %d", index, migraphx_->numTensors());
    return nullptr;
  }
  return migraphx_->tensor(index).name.c_str();
}

int32_t TrtCommon::getNbIOTensors() const
{
  if (!migraphx_) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Network is not initialized");
    return 0;
  }
  return migraphx_->numTensors();
}

nvinfer1::Dims TrtCommon::getTensorShape(const int32_t index) const
{
  if (!migraphx_) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Network is not initialized");
    return nvinfer1::Dims{};
  }
  if (migraphx_->engine) {
    return getTensorShape(getIOTensorName(index));
  }
  if (index < 0 || index >= migraphx_->numTensors()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR,
      "Invalid index for I/O tensor: %d. Total I/O tensors: %d", index, migraphx_->numTensors());
    return nvinfer1::Dims{};
  }
  return toDims(migraphx_->tensor(index).dims.value_or(std::vector<int64_t>{}));
}

nvinfer1::Dims TrtCommon::getTensorShape(const char * tensor_name) const
{
  if (!migraphx_ || !migraphx_->engine) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Engine is not initialized");
    return nvinfer1::Dims{};
  }
  const auto index = migraphx_->indexOf(tensor_name);
  if (!index) {
    return nvinfer1::Dims{-1, {}};
  }
  // Dimensions the model declares dynamic stay -1, as in a TensorRT engine; a tensor the model
  // gives no shape takes the shape of the compiled program.
  if (const auto & dims = migraphx_->tensor(*index).dims) {
    return toDims(*dims);
  }
  for (const auto & binding : migraphx_->engine->bindings) {
    if (binding.index == *index) {
      return toDims(binding.shape.lengths());
    }
  }
  return nvinfer1::Dims{-1, {}};
}

nvinfer1::TensorIOMode TrtCommon::getTensorIOMode(const char * tensor_name) const
{
  if (!migraphx_ || !migraphx_->engine) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Engine is not initialized");
    return nvinfer1::TensorIOMode::kNONE;
  }
  const auto index = migraphx_->indexOf(tensor_name);
  if (!index) {
    return nvinfer1::TensorIOMode::kNONE;
  }
  return *index < migraphx_->numInputs() ? nvinfer1::TensorIOMode::kINPUT
                                         : nvinfer1::TensorIOMode::kOUTPUT;
}

std::optional<nvinfer1::DataType> TrtCommon::getTensorDataType(const char * tensor_name) const
{
  if (!migraphx_ || !migraphx_->engine) {
    return std::nullopt;
  }
  const auto index = migraphx_->indexOf(tensor_name);
  if (!index) {
    return std::nullopt;
  }
  return toDataType(migraphx_->tensor(*index).elem_type);
}

nvinfer1::Dims TrtCommon::getInputDims(const int32_t index) const
{
  if (!migraphx_) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Network is not initialized");
    return {};
  }
  if (index < 0 || index >= migraphx_->numInputs()) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Invalid index for input tensor: %d", index);
    return {};
  }
  return toDims(migraphx_->model.inputs[index].dims.value_or(std::vector<int64_t>{}));
}

nvinfer1::Dims TrtCommon::getOutputDims(const int32_t index) const
{
  if (!migraphx_) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Network is not initialized");
    return {};
  }
  if (index < 0 || index >= static_cast<int32_t>(migraphx_->model.outputs.size())) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Invalid index for output tensor: %d", index);
    return {};
  }
  return toDims(migraphx_->model.outputs[index].dims.value_or(std::vector<int64_t>{}));
}

bool TrtCommon::setTensorAddress(const char * tensor_name, void * data)
{
  if (!migraphx_ || !migraphx_->engine) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Context is not initialized");
    return false;
  }
  if (tensor_name != nullptr && skipped_io_.count(tensor_name) > 0) {
    return true;
  }
  const auto index = migraphx_->indexOf(tensor_name);
  if (!index) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "Failed to set tensor address for tensor: %s",
      tensor_name ? tensor_name : "(null)");
    return false;
  }
  migraphx_->addresses[*index] = data;
  return true;
}

bool TrtCommon::setInputShape(const char * tensor_name, const nvinfer1::Dims & dimensions)
{
  if (!migraphx_ || !migraphx_->engine) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Context is not initialized");
    return false;
  }
  if (tensor_name != nullptr && skipped_io_.count(tensor_name) > 0) {
    return true;
  }
  const auto index = migraphx_->indexOf(tensor_name);
  auto valid = index && *index < migraphx_->numInputs() && dimensions.nbDims >= 0 &&
               dimensions.nbDims <= nvinfer1::Dims::MAX_DIMS;
  if (valid) {
    // As TensorRT does, accept only shapes the model and the optimization profile allow.
    const auto & declared = migraphx_->model.inputs[*index].dims;
    const ProfileDims * profile = nullptr;
    if (profile_dims_) {
      for (const auto & profile_dim : *profile_dims_) {
        if (profile_dim.tensor_name == tensor_name) {
          profile = &profile_dim;
        }
      }
    }
    valid = !declared || static_cast<int32_t>(declared->size()) == dimensions.nbDims;
    for (int32_t i = 0; valid && i < dimensions.nbDims; ++i) {
      const auto d = dimensions.d[i];
      valid = d > 0 && (!declared || (*declared)[i] < 0 || (*declared)[i] == d);
      if (valid && profile && i < profile->min_dims.nbDims && i < profile->max_dims.nbDims) {
        valid = profile->min_dims.d[i] <= d && d <= profile->max_dims.d[i];
      }
    }
  }
  if (!valid) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "Failed to set input shape %s for tensor: %s",
      TensorInfo::dimsRepr(dimensions).c_str(), tensor_name ? tensor_name : "(null)");
    return false;
  }
  migraphx_->input_shapes[*index] = Lens(dimensions.d, dimensions.d + dimensions.nbDims);
  return true;
}

std::shared_ptr<Profiler> TrtCommon::getModelProfiler()
{
  return model_profiler_;
}

std::shared_ptr<Profiler> TrtCommon::getHostProfiler()
{
  return host_profiler_;
}

std::shared_ptr<TrtCommonConfig> TrtCommon::getTrtCommonConfig()
{
  return trt_config_;
}

std::shared_ptr<nvinfer1::IBuilderConfig> TrtCommon::getBuilderConfig()
{
  logger_->log(
    nvinfer1::ILogger::Severity::kERROR,
    "MIGraphX has no TensorRT builder configuration; getBuilderConfig() returns null");
  return nullptr;
}

std::shared_ptr<nvinfer1::INetworkDefinition> TrtCommon::getNetwork()
{
  logger_->log(
    nvinfer1::ILogger::Severity::kERROR,
    "MIGraphX has no TensorRT network definition; getNetwork() returns null");
  return nullptr;
}

std::shared_ptr<Logger> TrtCommon::getLogger()
{
  return logger_;
}

bool TrtCommon::enqueueV3(cudaStream_t stream)
{
  if (!migraphx_ || !migraphx_->engine) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Context is not initialized");
    return false;
  }
  auto & state = *migraphx_;

  // The input shapes of this run; an input that was never given one keeps the shape `setup()`
  // compiled for.
  std::vector<Lens> input_lens(state.numInputs());
  for (int32_t i = 0; i < state.numInputs(); ++i) {
    input_lens[i] = state.input_shapes[i].value_or(state.engine->input_lens[i]);
  }

  auto & compiled = state.programs[input_lens];
  if (!compiled) {
    if (trt_config_->precision == "int8" && int8_calibrator_) {
      logger_->log(
        nvinfer1::ILogger::Severity::kERROR,
        "The int8 program is calibrated for the input shapes setup() compiled; run it with those");
      state.programs.erase(input_lens);
      return false;
    }
    std::string shapes;
    for (int32_t i = 0; i < state.numInputs(); ++i) {
      shapes += (i ? ", " : "") + state.model.inputs[i].name + " " + lensRepr(input_lens[i]);
    }
    logger_->log(
      nvinfer1::ILogger::Severity::kINFO,
      "Compiling the model for new input shapes (%s); each new shape is compiled once",
      shapes.c_str());
    try {
      const ProgramBuilder builder{*trt_config_, state.model, nullptr, *logger_};
      compiled = describeProgram(builder.compile(input_lens), state.model);
    } catch (const std::exception & e) {
      logger_->log(
        nvinfer1::ILogger::Severity::kERROR, "Fail to compile for new input shapes: %s", e.what());
      state.programs.erase(input_lens);
      return false;
    }
  }

  migraphx::program_parameters parameters;
  for (const auto & binding : compiled->bindings) {
    void * address = state.addresses[binding.index];
    if (address == nullptr) {
      logger_->log(
        nvinfer1::ILogger::Severity::kERROR, "Tensor address is not set for tensor: %s",
        state.tensor(binding.index).name.c_str());
      return false;
    }
    parameters.add(binding.name.c_str(), migraphx::argument(binding.shape, address));
  }

  const auto run = [&]() -> bool {
    try {
      compiled->program.run_async(parameters, stream);
    } catch (const std::exception & e) {
      logger_->log(nvinfer1::ILogger::Severity::kERROR, "Inference failed: %s", e.what());
      return false;
    }
    return true;
  };
  if (trt_config_->profile_per_layer) {
    auto inference_start = std::chrono::high_resolution_clock::now();
    auto success = run();
    auto inference_end = std::chrono::high_resolution_clock::now();
    host_profiler_->reportLayerTime(
      "inference_host",
      std::chrono::duration<float, std::milli>(inference_end - inference_start).count());
    return success;
  }
  return run();
}

void TrtCommon::printProfiling() const
{
  logger_->log(
    nvinfer1::ILogger::Severity::kINFO, "Host Profiling\n%s", host_profiler_->toString().c_str());
  logger_->log(
    nvinfer1::ILogger::Severity::kINFO, "Model Profiling\n%s", model_profiler_->toString().c_str());
}

std::string TrtCommon::getLayerInformation(nvinfer1::LayerInformationFormat format)
{
  (void)format;
  logger_->log(
    nvinfer1::ILogger::Severity::kWARNING, "MIGraphX provides no per-layer engine information");
  return {};
}

bool TrtCommon::isStronglyTyped() const
{
  return trt_config_->precision == "strongly-typed";
}

bool TrtCommon::buildEngineFromOnnx()
{
  auto & state = *migraphx_;

  // Compile for the optimal profile shapes; inputs without a profile take the model's shape,
  // with any dynamic dimension set to 1.
  std::vector<std::string> profile_lines;
  if (profile_dims_) {
    for (const auto & profile_dim : *profile_dims_) {
      logger_->log(
        nvinfer1::ILogger::Severity::kINFO, "Setting optimization profile for tensor: %s",
        profile_dim.toString().c_str());
      profile_lines.push_back(profile_dim.toString());
    }
  }
  std::vector<Lens> input_lens;
  for (const auto & input : state.model.inputs) {
    const ProfileDims * profile = nullptr;
    if (profile_dims_) {
      for (const auto & profile_dim : *profile_dims_) {
        if (profile_dim.tensor_name == input.name) {
          profile = &profile_dim;
        }
      }
    }
    Lens lens;
    if (profile) {
      lens.assign(profile->opt_dims.d, profile->opt_dims.d + profile->opt_dims.nbDims);
    } else {
      for (const auto d : input.dims.value_or(std::vector<int64_t>{})) {
        lens.push_back(d > 0 ? static_cast<std::size_t>(d) : 1U);
      }
    }
    input_lens.push_back(std::move(lens));
  }

  std::shared_ptr<CompiledProgram> compiled;
  try {
    const ProgramBuilder builder{*trt_config_, state.model, int8_calibrator_, *logger_};
    compiled = describeProgram(builder.compile(input_lens), state.model);
  } catch (const std::exception & e) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Fail to build engine: %s", e.what());
    return false;
  }

  // Save the program and the key it is valid for
  const auto program_path = migraphxProgramPath(*trt_config_, state.arch);
  std::ofstream key_file(program_path.string() + ".key", std::ios::trunc);
  if (!key_file.is_open()) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Fail to open engine file");
    return false;
  }
  try {
    migraphx::save(compiled->program, program_path.c_str());
  } catch (const std::exception & e) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Fail to save engine: %s", e.what());
    return false;
  }
  for (const auto & line : cacheKey(state.arch, trt_config_->precision)) {
    key_file << line << '\n';
  }
  for (const auto & line : profile_lines) {
    key_file << kProfilePrefix << line << '\n';
  }

  state.programs.clear();
  state.programs[compiled->input_lens] = compiled;
  state.engine = compiled;
  state.engine_profile = std::set<std::string>(profile_lines.begin(), profile_lines.end());
  return true;
}

bool TrtCommon::validateEngine()
{
  const auto program_path = migraphxProgramPath(*trt_config_, migraphx_->arch);
  std::ifstream key_file(program_path.string() + ".key");
  std::vector<std::string> key;
  for (std::string line; std::getline(key_file, line) && line.rfind(kProfilePrefix, 0) != 0;) {
    key.push_back(line);
  }
  const auto expected = cacheKey(migraphx_->arch, trt_config_->precision);
  logger_->log(
    nvinfer1::ILogger::Severity::kINFO, "Program was compiled for: %s",
    key.empty() ? "(no key)" : key.front().c_str());
  if (key != expected) {
    logger_->log(
      nvinfer1::ILogger::Severity::kWARNING,
      "Program was compiled with a different MIGraphX, GPU or precision! Current: %s, %s, %s",
      expected[0].c_str(), expected[1].c_str(), expected[2].c_str());
    return false;
  }
  return true;
}

bool TrtCommon::loadEngine()
{
  auto & state = *migraphx_;
  const auto program_path = migraphxProgramPath(*trt_config_, state.arch);
  std::shared_ptr<CompiledProgram> compiled;
  try {
    compiled = describeProgram(migraphx::load(program_path.c_str()), state.model);
  } catch (const std::exception & e) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR, "Fail to create engine: %s", e.what());
    return false;
  }

  std::set<std::string> profile;
  std::ifstream key_file(program_path.string() + ".key");
  for (std::string line; std::getline(key_file, line);) {
    if (line.rfind(kProfilePrefix, 0) == 0) {
      profile.insert(line.substr(std::strlen(kProfilePrefix)));
    }
  }

  state.programs.clear();
  state.programs[compiled->input_lens] = compiled;
  state.engine = compiled;
  state.engine_profile = std::move(profile);
  return true;
}

bool TrtCommon::validateProfileDims()
{
  auto success = true;
  if (!profile_dims_ || profile_dims_->empty()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kWARNING,
      "Input profile is empty, skipping validation. If network has dynamic shapes, it might lead "
      "to undefined behavior.");
    return success;
  }
  for (const auto & profile_dim : *profile_dims_) {
    if (migraphx_->engine_profile.count(profile_dim.toString()) == 0) {
      logger_->log(
        nvinfer1::ILogger::Severity::kWARNING,
        "Invalid profile. Current configuration: %s. Not the profile of the cached engine.",
        profile_dim.toString().c_str());
      success = false;
    }
  }
  return success;
}

}  // namespace tensorrt_common
}  // namespace autoware
