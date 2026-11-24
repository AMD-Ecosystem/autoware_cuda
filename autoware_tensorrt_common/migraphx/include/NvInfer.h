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

// Stands in for TensorRT's NvInfer.h on ROCm builds, where autoware_tensorrt_common runs models
// with MIGraphX and this directory is on the include path. It declares only the value types and
// interfaces used by this package's headers and by the code that includes them; nothing that
// builds or runs a TensorRT engine is declared.

#ifndef NVINFER_H_
#define NVINFER_H_

// NV_INFER_H and NV_INFER_VERSION_H are the include guards of TensorRT's NvInfer.h and
// NvInferVersion.h, which every TensorRT header includes.
#if defined(NV_INFER_H) || defined(NV_INFER_VERSION_H)
#error "TensorRT headers are included together with the MIGraphX build of autoware_tensorrt_common"
#endif

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Lets the package headers select their MIGraphX declarations. The library target also defines
// it, so that code reaching real TensorRT headers first stops at the check above.
#ifndef AUTOWARE_TENSORRT_COMMON_MIGRAPHX
#define AUTOWARE_TENSORRT_COMMON_MIGRAPHX 1
#endif

#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif

// The TensorRT API level these declarations follow (TensorRT 10, with 64-bit dimensions), for code
// that prints or compares the TensorRT version. This is not a TensorRT version: no TensorRT is
// used.
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 0
#define NV_TENSORRT_PATCH 0

namespace nvinfer1
{
class Dims64
{
public:
  static constexpr int32_t MAX_DIMS{8};
  int32_t nbDims;
  int64_t d[MAX_DIMS];
};

using Dims = Dims64;

class Dims2 : public Dims
{
public:
  Dims2() : Dims2(0, 0) {}
  Dims2(int64_t d0, int64_t d1) : Dims{2, {d0, d1}} {}
};

class Dims3 : public Dims
{
public:
  Dims3() : Dims3(0, 0, 0) {}
  Dims3(int64_t d0, int64_t d1, int64_t d2) : Dims{3, {d0, d1, d2}} {}
};

class Dims4 : public Dims
{
public:
  Dims4() : Dims4(0, 0, 0, 0) {}
  Dims4(int64_t d0, int64_t d1, int64_t d2, int64_t d3) : Dims{4, {d0, d1, d2, d3}} {}
};

enum class DataType : int32_t {
  kFLOAT = 0,
  kHALF = 1,
  kINT8 = 2,
  kINT32 = 3,
  kBOOL = 4,
  kUINT8 = 5,
  kFP8 = 6,
  kBF16 = 7,
  kINT64 = 8,
  kINT4 = 9,
};

enum class TensorIOMode : int32_t {
  kNONE = 0,
  kINPUT = 1,
  kOUTPUT = 2,
};

enum class LayerType : int32_t {
  kCONVOLUTION,
  kACTIVATION,
  kPOOLING,
  kSOFTMAX,
  kMATRIX_MULTIPLY,
  kRESIZE,
  kCONSTANT,
};

enum class ActivationType : int32_t {
  kRELU = 0,
  kSIGMOID = 1,
  kTANH = 2,
};

enum class LayerInformationFormat : int32_t {
  kONELINE = 0,
  kJSON = 1,
};

enum class CalibrationAlgoType : int32_t {
  kLEGACY_CALIBRATION = 0,
  kENTROPY_CALIBRATION = 1,
  kENTROPY_CALIBRATION_2 = 2,
  kMINMAX_CALIBRATION = 3,
};

class ILogger
{
public:
  enum class Severity : int32_t {
    kINTERNAL_ERROR = 0,
    kERROR = 1,
    kWARNING = 2,
    kINFO = 3,
    kVERBOSE = 4,
  };

  virtual void log(Severity severity, const char * msg) noexcept = 0;

  virtual ~ILogger() = default;
};

class IProfiler
{
public:
  virtual void reportLayerTime(const char * layerName, float ms) noexcept = 0;

  virtual ~IProfiler() = default;
};

class IInt8Calibrator
{
public:
  virtual int32_t getBatchSize() const noexcept = 0;
  virtual bool getBatch(void * bindings[], const char * names[], int32_t nbBindings) noexcept = 0;
  virtual const void * readCalibrationCache(std::size_t & length) noexcept = 0;
  virtual void writeCalibrationCache(const void * ptr, std::size_t length) noexcept = 0;
  virtual CalibrationAlgoType getAlgorithm() noexcept = 0;

  virtual ~IInt8Calibrator() = default;
};

class IInt8EntropyCalibrator : public IInt8Calibrator
{
public:
  CalibrationAlgoType getAlgorithm() noexcept override
  {
    return CalibrationAlgoType::kENTROPY_CALIBRATION;
  }
};

class IInt8EntropyCalibrator2 : public IInt8Calibrator
{
public:
  CalibrationAlgoType getAlgorithm() noexcept override
  {
    return CalibrationAlgoType::kENTROPY_CALIBRATION_2;
  }
};

class IInt8MinMaxCalibrator : public IInt8Calibrator
{
public:
  CalibrationAlgoType getAlgorithm() noexcept override
  {
    return CalibrationAlgoType::kMINMAX_CALIBRATION;
  }
};

class IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
  CalibrationAlgoType getAlgorithm() noexcept override
  {
    return CalibrationAlgoType::kLEGACY_CALIBRATION;
  }
  virtual double getQuantile() const noexcept = 0;
  virtual double getRegressionCutoff() const noexcept = 0;
  virtual const void * readHistogramCache(std::size_t & length) noexcept = 0;
  virtual void writeHistogramCache(const void * ptr, std::size_t length) noexcept = 0;
};

// MIGraphX exposes no network layers, so no object of these types ever exists here. They are
// declared for code that inspects TensorRT layers and has to compile on both backends.
class ITensor
{
public:
  virtual const char * getName() const noexcept = 0;
  virtual Dims getDimensions() const noexcept = 0;

protected:
  virtual ~ITensor() = default;
};

class ILayer
{
public:
  virtual LayerType getType() const noexcept = 0;
  virtual const char * getName() const noexcept = 0;
  virtual ITensor * getInput(int32_t index) const noexcept = 0;
  virtual ITensor * getOutput(int32_t index) const noexcept = 0;
  virtual void setPrecision(DataType dataType) noexcept = 0;

protected:
  virtual ~ILayer() = default;
};

class IConvolutionLayer : public ILayer
{
public:
  virtual Dims getKernelSizeNd() const noexcept = 0;
  virtual Dims getStrideNd() const noexcept = 0;
  virtual int64_t getNbGroups() const noexcept = 0;
};

class IActivationLayer : public ILayer
{
public:
  virtual ActivationType getActivationType() const noexcept = 0;
};

// getNetwork() and getBuilderConfig() return null with MIGraphX, so these are never instantiated
// either; code that checks for null before using them compiles on both backends.
class INetworkDefinition
{
public:
  virtual int32_t getNbLayers() const noexcept = 0;
  virtual ILayer * getLayer(int32_t index) const noexcept = 0;

  virtual ~INetworkDefinition() = default;
};

class IBuilderConfig
{
public:
  virtual void setMaxAuxStreams(int32_t nbStreams) noexcept = 0;

  virtual ~IBuilderConfig() = default;
};
}  // namespace nvinfer1

#endif  // NVINFER_H_
