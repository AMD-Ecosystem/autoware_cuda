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

// Behavior of the MIGraphX backend that the shared IO tests do not reach: inference results,
// IO order, dynamic shapes, precisions, the compiled program cache, and plugin libraries.

#include <autoware/cuda_utils/cuda_gtest_utils.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <autoware/tensorrt_common/tensorrt_conv_calib.hpp>
#include <autoware/tensorrt_common/utils.hpp>
#include <migraphx/migraphx.hpp>

#include <NvInfer.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using autoware::tensorrt_common::CalibrationConfig;
using autoware::tensorrt_common::NetworkIO;
using autoware::tensorrt_common::ProfileDims;
using autoware::tensorrt_common::TrtCommon;
using autoware::tensorrt_common::TrtCommonConfig;
using autoware::tensorrt_common::TrtConvCalib;

const std::filesystem::path kDataDir{TEST_DATA_DIR};

// conv_network.onnx: image [batch, 3, 8, 8] -> logits [batch, 2], features [batch, 4].
constexpr int64_t kChannels = 3;
constexpr int64_t kSize = 8;
constexpr int64_t kClasses = 2;
constexpr int64_t kFeatures = 4;
constexpr int64_t kReferenceBatch = 16;
const nvinfer1::Dims kImageMin{4, {1, kChannels, kSize, kSize}};
const nvinfer1::Dims kImageOpt{4, {8, kChannels, kSize, kSize}};
const nvinfer1::Dims kImageMax{4, {kReferenceBatch, kChannels, kSize, kSize}};

nvinfer1::Dims imageDims(const int64_t batch, const int64_t size = kSize)
{
  return nvinfer1::Dims{4, {batch, kChannels, size, size}};
}

void expectDims(const nvinfer1::Dims & actual, const std::vector<int64_t> & expected)
{
  ASSERT_EQ(actual.nbDims, static_cast<int32_t>(expected.size()));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.d[i], expected[i]) << "dimension " << i;
  }
}

std::vector<float> readFloats(const std::filesystem::path & path)
{
  std::ifstream file(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::vector<float> values(bytes.size() / sizeof(float));
  std::copy_n(bytes.data(), values.size() * sizeof(float), reinterpret_cast<char *>(values.data()));
  return values;
}

/// Element i of the test input, as generate_conv_network.py builds it.
float inputValue(const std::size_t i)
{
  return static_cast<float>((i * 37) % 101) / 101.0F - 0.5F;
}

#define ASSERT_HIP_SUCCESS(expr) ASSERT_EQ((expr), hipSuccess)

/// A device buffer of floats.
class DeviceBuffer
{
public:
  explicit DeviceBuffer(const std::size_t count) : count_(count)
  {
    if (hipMalloc(&data_, count * sizeof(float)) != hipSuccess) {
      throw std::runtime_error("hipMalloc failed");
    }
  }
  ~DeviceBuffer() { (void)hipFree(data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer & operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] void * get() const { return data_; }

  void upload(const std::vector<float> & values) const
  {
    if (
      hipMemcpy(data_, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice) !=
      hipSuccess) {
      throw std::runtime_error("hipMemcpy failed");
    }
  }

  [[nodiscard]] std::vector<float> download() const
  {
    std::vector<float> values(count_);
    if (
      hipMemcpy(values.data(), data_, count_ * sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
      throw std::runtime_error("hipMemcpy failed");
    }
    return values;
  }

private:
  void * data_{nullptr};
  std::size_t count_;
};

/// Whether MIGraphX itself compiles a test model with `input` of the shape `lens`.
bool compilesFor(
  const std::string & model, const std::string & input, std::vector<std::size_t> lens)
{
  try {
    migraphx::onnx_options options;
    options.set_input_parameter_shape(input, std::move(lens));
    auto program = migraphx::parse_onnx((kDataDir / model).c_str(), options);
    migraphx::compile_options compile_options;
    compile_options.set_offload_copy(false);
    program.compile(migraphx::target("gpu"), compile_options);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

/// Feeds the test input as calibration batches and records how it is used.
class TestCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
  TestCalibrator(const int64_t batch, const int32_t num_batches)
  : batch_(batch), num_batches_(num_batches), buffer_(batch * kChannels * kSize * kSize)
  {
  }

  int32_t getBatchSize() const noexcept override { return static_cast<int32_t>(batch_); }

  bool getBatch(void * bindings[], const char * names[], int32_t nb_bindings) noexcept override
  {
    if (batches_served_ == num_batches_ || nb_bindings != 1 || std::string(names[0]) != "image") {
      return false;
    }
    std::vector<float> values(batch_ * kChannels * kSize * kSize);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = inputValue(i + batches_served_ * 7);
    }
    try {
      buffer_.upload(values);
    } catch (const std::exception &) {
      return false;
    }
    bindings[0] = buffer_.get();
    ++batches_served_;
    return true;
  }

  const void * readCalibrationCache(std::size_t & length) noexcept override
  {
    length = 0;
    return nullptr;
  }

  void writeCalibrationCache(const void *, std::size_t) noexcept override { ++cache_writes_; }

  int32_t batches_served_{0};
  int32_t cache_writes_{0};

private:
  int64_t batch_;
  int32_t num_batches_;
  DeviceBuffer buffer_;
};
}  // namespace

class MigraphxCommonTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    SKIP_TEST_IF_CUDA_UNAVAILABLE();

    // A directory of its own, so that test processes running at the same time do not share it.
    auto pattern =
      (std::filesystem::temp_directory_path() /
       ("autoware_tensorrt_common_migraphx_test_" +
        std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "_XXXXXX"))
        .string();
    ASSERT_NE(mkdtemp(pattern.data()), nullptr) << "cannot create " << pattern;
    work_dir_ = pattern;
    ASSERT_HIP_SUCCESS(hipStreamCreate(&stream_));

    reference_ = readFloats(kDataDir / "conv_network_reference.bin");
    ASSERT_EQ(reference_.size(), kReferenceBatch * (kClasses + kFeatures))
      << "missing conv_network_reference.bin";
  }

  void TearDown() override
  {
    if (stream_) {
      (void)hipStreamDestroy(stream_);
    }
    std::error_code ignored;
    std::filesystem::remove_all(work_dir_, ignored);
  }

  [[nodiscard]] std::string enginePath(const std::string & tag) const
  {
    return (work_dir_ / (tag + ".engine")).string();
  }

  std::unique_ptr<TrtCommon> makeConv(const std::string & precision, const std::string & tag)
  {
    return std::make_unique<TrtCommon>(TrtCommonConfig(
      (kDataDir / "conv_network.onnx").string(), precision, enginePath(tag), 1ULL << 26U));
  }

  static std::vector<ProfileDims> convProfile()
  {
    return {ProfileDims("image", kImageMin, kImageOpt, kImageMax)};
  }

  static std::vector<NetworkIO> convNetworkIO(
    const std::string & input = "image", const int64_t size = kSize)
  {
    return {
      NetworkIO(input, {4, {-1, kChannels, size, size}}), NetworkIO("logits", {2, {-1, kClasses}}),
      NetworkIO("features", {2, {-1, kFeatures}})};
  }

  static bool setupConv(TrtCommon & trt, const std::string & input = "image", int64_t size = kSize)
  {
    return trt.setup(
      std::make_unique<std::vector<ProfileDims>>(std::vector<ProfileDims>{ProfileDims(
        input, imageDims(1, size), imageDims(8, size), imageDims(kReferenceBatch, size))}),
      std::make_unique<std::vector<NetworkIO>>(convNetworkIO(input, size)));
  }

  /// A TrtCommon for the model at `model`, a copy that tests replace, with its program cached in
  /// `dir`.
  static std::unique_ptr<TrtCommon> makeAt(
    const std::filesystem::path & dir, const std::filesystem::path & model,
    const std::string & precision = "fp32")
  {
    return std::make_unique<TrtCommon>(
      TrtCommonConfig(model.string(), precision, (dir / "model.engine").string(), 1ULL << 26U));
  }

  /// Copy a test model to `model`, replacing what is there.
  static void placeModel(const std::string & name, const std::filesystem::path & model)
  {
    std::filesystem::create_directories(model.parent_path());
    std::filesystem::copy_file(
      kDataDir / name, model, std::filesystem::copy_options::overwrite_existing);
  }

  [[nodiscard]] std::filesystem::path programIn(const std::filesystem::path & dir) const
  {
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() == ".mxr") {
        return entry.path();
      }
    }
    return {};
  }

  [[nodiscard]] std::vector<std::filesystem::path> programFiles() const
  {
    std::vector<std::filesystem::path> files;
    for (const auto & entry : std::filesystem::directory_iterator(work_dir_)) {
      if (entry.path().extension() == ".mxr") {
        files.push_back(entry.path());
      }
    }
    return files;
  }

  /// Run a model of x [rows, 4] and one output y at `rows` rows, with y in a buffer sized from
  /// getOutputDims(0), each -1 taken as the length `known` gives for that dimension, and a guard
  /// region after it, which a run must not write. Returns what the run wrote in the sized buffer.
  std::vector<float> runRows(
    TrtCommon & trt, const int64_t rows, const std::vector<int64_t> & known,
    const std::vector<std::pair<std::string, float>> & scalars = {})
  {
    static constexpr std::size_t kGuard = 1024;
    static constexpr float kSentinel = -12345.0F;
    const auto reported = trt.getOutputDims(0);
    std::size_t size = 1;
    for (int32_t i = 0; i < reported.nbDims; ++i) {
      size *= static_cast<std::size_t>(reported.d[i] < 0 ? known.at(i) : reported.d[i]);
    }
    std::vector<float> x(rows * 4);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = inputValue(i);
    }
    DeviceBuffer x_d(x.size());
    DeviceBuffer y_d(size + kGuard);
    x_d.upload(x);
    y_d.upload(std::vector<float>(size + kGuard, kSentinel));
    EXPECT_TRUE(trt.setTensor("x", x_d.get(), {2, {rows, 4}}));
    std::vector<std::unique_ptr<DeviceBuffer>> scalar_buffers;
    for (const auto & [name, value] : scalars) {
      auto & buffer = scalar_buffers.emplace_back(std::make_unique<DeviceBuffer>(1));
      buffer->upload({value});
      EXPECT_TRUE(trt.setTensor(name.c_str(), buffer->get(), {0, {}}));
    }
    EXPECT_TRUE(trt.setTensorAddress("y", y_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_)) << "rows " << rows;
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto y = y_d.download();
    EXPECT_TRUE(
      std::all_of(
        y.begin() + static_cast<std::ptrdiff_t>(size), y.end(),
        [](float v) { return v == kSentinel; }))
      << "a run at " << rows << " rows wrote past a buffer sized from the reported shape";
    y.resize(size);
    return y;
  }

  /// Check `y` against the ReLU of the input x of `rows` rows.
  static void expectRelu(const std::vector<float> & y, const int64_t rows)
  {
    ASSERT_EQ(y.size(), static_cast<std::size_t>(rows * 4)) << "rows " << rows;
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_EQ(y[i], std::max(inputValue(i), 0.0F)) << "rows " << rows << ", element " << i;
    }
  }

  /// Run the conv network on `batch` samples and return the largest error against the
  /// reference, with the outputs read from the buffers the caller bound.
  float runConv(
    TrtCommon & trt, const int64_t batch, const std::string & input = "image",
    const int64_t size = kSize, const std::vector<float> * reference = nullptr)
  {
    if (reference == nullptr) {
      reference = &reference_;
    }
    std::vector<float> image(batch * kChannels * size * size);
    for (std::size_t i = 0; i < image.size(); ++i) {
      image[i] = inputValue(i);
    }
    DeviceBuffer image_d(image.size());
    DeviceBuffer logits_d(batch * kClasses);
    DeviceBuffer features_d(batch * kFeatures);
    image_d.upload(image);
    // Outputs that never get written stay NaN and fail the comparison.
    logits_d.upload(std::vector<float>(batch * kClasses, std::nanf("")));
    features_d.upload(std::vector<float>(batch * kFeatures, std::nanf("")));

    EXPECT_TRUE(trt.setTensor(input.c_str(), image_d.get(), imageDims(batch, size)));
    EXPECT_TRUE(trt.setTensorAddress("logits", logits_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("features", features_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    const auto logits = logits_d.download();
    const auto features = features_d.download();
    const float * reference_logits = reference->data();
    const float * reference_features = reference->data() + kReferenceBatch * kClasses;
    float max_error = 0.0F;
    const auto compare = [&max_error](float actual, float expected) {
      const auto error = std::fabs(actual - expected);
      max_error =
        std::isnan(error) ? std::numeric_limits<float>::infinity() : std::max(max_error, error);
    };
    for (std::size_t i = 0; i < logits.size(); ++i) {
      compare(logits[i], reference_logits[i]);
    }
    for (std::size_t i = 0; i < features.size(); ++i) {
      compare(features[i], reference_features[i]);
    }
    return max_error;
  }

  std::filesystem::path work_dir_;
  hipStream_t stream_{nullptr};
  std::vector<float> reference_;
};

TEST_F(MigraphxCommonTest, IOTensorsFollowTheModelOrder)
{
  auto trt = makeConv("fp32", "io_order");
  ASSERT_EQ(trt->getNbIOTensors(), 3);
  EXPECT_STREQ(trt->getIOTensorName(0), "image");
  EXPECT_STREQ(trt->getIOTensorName(1), "logits");
  EXPECT_STREQ(trt->getIOTensorName(2), "features");
  EXPECT_EQ(trt->getIOTensorName(3), nullptr);
  EXPECT_EQ(trt->getInputDims(0).nbDims, 4);
  EXPECT_EQ(trt->getInputDims(0).d[0], -1);
  EXPECT_EQ(trt->getOutputDims(1).d[1], kFeatures);

  ASSERT_TRUE(setupConv(*trt));
  EXPECT_STREQ(trt->getIOTensorName(1), "logits");
  EXPECT_EQ(trt->getTensorIOMode("image"), nvinfer1::TensorIOMode::kINPUT);
  EXPECT_EQ(trt->getTensorIOMode("features"), nvinfer1::TensorIOMode::kOUTPUT);
  EXPECT_EQ(trt->getTensorIOMode("absent"), nvinfer1::TensorIOMode::kNONE);
  EXPECT_EQ(trt->getTensorDataType("logits"), nvinfer1::DataType::kFLOAT);
  const auto image = trt->getTensorShape("image");
  ASSERT_EQ(image.nbDims, 4);
  EXPECT_EQ(image.d[0], -1);
  EXPECT_EQ(image.d[3], kSize);
}

TEST_F(MigraphxCommonTest, Fp32InferenceWritesTheCallerBuffers)
{
  auto trt = makeConv("fp32", "fp32");
  ASSERT_TRUE(setupConv(*trt));
  EXPECT_LT(runConv(*trt, 8), 1e-5F);
}

TEST_F(MigraphxCommonTest, DynamicBatchRunsAcrossTheProfile)
{
  auto trt = makeConv("fp32", "dynamic");
  ASSERT_TRUE(setupConv(*trt));
  for (const int64_t batch : std::vector<int64_t>{8, 1, kReferenceBatch, 3, 8}) {
    EXPECT_LT(runConv(*trt, batch), 1e-5F) << "batch " << batch;
  }
  EXPECT_FALSE(trt->setInputShape("image", imageDims(kReferenceBatch + 1)));
  EXPECT_FALSE(trt->setInputShape("image", {4, {1, kChannels, kSize, kSize + 1}}));
  EXPECT_FALSE(trt->setInputShape("logits", {2, {1, kClasses}}));
}

TEST_F(MigraphxCommonTest, Fp16InferenceIsCloseToFp32)
{
  auto trt = makeConv("fp16", "fp16");
  ASSERT_TRUE(setupConv(*trt));
  const auto quantized = runConv(*trt, 8);
  EXPECT_LT(quantized, 1e-2F);
  ASSERT_EQ(programFiles().size(), 1U);
  EXPECT_NE(programFiles().front().filename().string().find(".fp16.mxr"), std::string::npos);

  // Excluding every computing operation keeps fp32 accuracy, also in the program enqueueV3()
  // compiles for a new batch size.
  TrtCommonConfig config(
    (kDataDir / "conv_network.onnx").string(), "fp16", enginePath("fp16_excluded"), 1ULL << 26U);
  config.fp16_excluded_ops = {"add", "convolution", "dot", "pooling", "relu"};
  TrtCommon excluded(config);
  ASSERT_TRUE(setupConv(excluded));
  const auto kept = runConv(excluded, 8);
  std::cout << "largest error: " << quantized << " in fp16, " << kept
            << " with every operation kept" << std::endl;
  EXPECT_GT(quantized, 10.0F * kept) << "fp16 was not applied, or not undone";
  EXPECT_LT(kept, 1e-5F);
  EXPECT_LT(runConv(excluded, 3), 1e-5F);
}

TEST_F(MigraphxCommonTest, Int8CalibratesFromTheCalibratorBatches)
{
  TrtConvCalib trt(TrtCommonConfig(
    (kDataDir / "conv_network.onnx").string(), "int8", enginePath("int8"), 1ULL << 26U));
  auto calibrator = std::make_unique<TestCalibrator>(8, 4);
  const auto * observed = calibrator.get();
  ASSERT_TRUE(trt.setupWithCalibrator(
    std::move(calibrator), CalibrationConfig("Entropy", false, false, 0.0),
    std::make_unique<std::vector<ProfileDims>>(convProfile()),
    std::make_unique<std::vector<NetworkIO>>(convNetworkIO())));
  EXPECT_EQ(observed->batches_served_, 4);
  // A TensorRT calibration cache written from MIGraphX would mislead a TensorRT build.
  EXPECT_EQ(observed->cache_writes_, 0);
  // Outputs are within about 0.5 here; int8 keeps them within a few percent of that.
  EXPECT_LT(runConv(trt, 8), 5e-2F);
  // The calibration holds for the shapes it was made with only.
  EXPECT_TRUE(trt.setInputShape("image", imageDims(2)));
  EXPECT_FALSE(trt.enqueueV3(stream_));
}

TEST_F(MigraphxCommonTest, CachedProgramIsLoadedAndRebuiltWhenItsKeyChanges)
{
  {
    auto trt = makeConv("fp32", "cache");
    ASSERT_TRUE(setupConv(*trt));
  }
  const auto files = programFiles();
  ASSERT_EQ(files.size(), 1U);
  const auto program = files.front();
  const auto key = std::filesystem::path(program.string() + ".key");
  ASSERT_TRUE(std::filesystem::exists(key));
  EXPECT_FALSE(std::filesystem::exists(enginePath("cache")))
    << "the TensorRT engine path is left for TensorRT";
  const auto built = std::filesystem::last_write_time(program);

  {
    auto trt = makeConv("fp32", "cache");
    ASSERT_TRUE(setupConv(*trt));
    EXPECT_EQ(std::filesystem::last_write_time(program), built) << "the program was rebuilt";
    EXPECT_LT(runConv(*trt, 8), 1e-5F);
  }

  // A program compiled by another MIGraphX is rebuilt.
  std::ifstream key_in(key);
  std::string key_text((std::istreambuf_iterator<char>(key_in)), std::istreambuf_iterator<char>());
  key_in.close();
  std::ofstream(key, std::ios::trunc) << "migraphx 0.0.0\n"
                                      << key_text.substr(key_text.find('\n') + 1);
  std::filesystem::last_write_time(program, built - std::chrono::hours(1));
  {
    auto trt = makeConv("fp32", "cache");
    ASSERT_TRUE(setupConv(*trt));
    EXPECT_GT(std::filesystem::last_write_time(program), built - std::chrono::hours(1));
    EXPECT_LT(runConv(*trt, 8), 1e-5F);
  }

  // So is one built for another optimization profile.
  const auto rebuilt = std::filesystem::last_write_time(program);
  std::filesystem::last_write_time(program, rebuilt - std::chrono::hours(1));
  {
    auto trt = makeConv("fp32", "cache");
    ASSERT_TRUE(trt->setup(
      std::make_unique<std::vector<ProfileDims>>(
        std::vector<ProfileDims>{ProfileDims("image", kImageMin, imageDims(4), kImageMax)})));
    EXPECT_GT(std::filesystem::last_write_time(program), rebuilt - std::chrono::hours(1));
    EXPECT_LT(runConv(*trt, 4), 1e-5F);
  }
}

TEST_F(MigraphxCommonTest, DummyNetworkAddsItsThreeInputs)
{
  auto trt = std::make_unique<TrtCommon>(TrtCommonConfig(
    (kDataDir / "dummy_network.onnx").string(), "fp32", enginePath("dummy"), 1ULL << 26U));
  const nvinfer1::Dims shape{2, {-1, 4}};
  ASSERT_TRUE(trt->setup(
    std::make_unique<std::vector<ProfileDims>>(std::vector<ProfileDims>{
      ProfileDims("required_a", {2, {1, 4}}, {2, {8, 4}}, {2, {16, 4}}),
      ProfileDims("required_b", {2, {1, 4}}, {2, {8, 4}}, {2, {16, 4}}),
      ProfileDims("optional_present", {2, {1, 4}}, {2, {8, 4}}, {2, {16, 4}})}),
    std::make_unique<std::vector<NetworkIO>>(std::vector<NetworkIO>{
      NetworkIO("required_a", shape), NetworkIO("required_b", shape),
      NetworkIO("optional_present", shape), NetworkIO("sum", shape)})));

  constexpr int64_t kRows = 5;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  for (int i = 0; i < 4; ++i) {
    buffers.push_back(std::make_unique<DeviceBuffer>(kRows * 4));
  }
  std::vector<void *> addresses;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> values(kRows * 4);
    for (std::size_t j = 0; j < values.size(); ++j) {
      values[j] = i == 3 ? std::nanf("") : static_cast<float>((i + 1) * 100 + j);
    }
    buffers[i]->upload(values);
    addresses.push_back(buffers[i]->get());
  }
  ASSERT_TRUE(trt->setTensorsAddresses(addresses));
  ASSERT_TRUE(trt->setInputsShapes({{2, {kRows, 4}}, {2, {kRows, 4}}, {2, {kRows, 4}}}));
  ASSERT_TRUE(trt->enqueueV3(stream_));
  ASSERT_HIP_SUCCESS(hipStreamSynchronize(stream_));
  const auto sum = buffers[3]->download();
  for (std::size_t j = 0; j < sum.size(); ++j) {
    EXPECT_FLOAT_EQ(sum[j], 600.0F + 3.0F * static_cast<float>(j)) << "element " << j;
  }
}

TEST_F(MigraphxCommonTest, RefusedSetupCompilesNothing)
{
  auto trt = makeConv("fp32", "refused");
  auto io = convNetworkIO();
  io.emplace_back("required_missing", nvinfer1::Dims{1, {1}});
  EXPECT_FALSE(trt->setup(
    std::make_unique<std::vector<ProfileDims>>(convProfile()),
    std::make_unique<std::vector<NetworkIO>>(std::move(io))));
  EXPECT_TRUE(programFiles().empty());
}

TEST_F(MigraphxCommonTest, TensorRTOnlyFeaturesAreRefused)
{
  auto trt = makeConv("fp32", "unsupported");
  EXPECT_EQ(trt->getNetwork(), nullptr);
  EXPECT_EQ(trt->getBuilderConfig(), nullptr);
  EXPECT_FALSE(trt->enqueueV3(stream_)) << "no program before setup()";
  EXPECT_THROW(
    TrtCommon(TrtCommonConfig((kDataDir / "conv_network.onnx").string(), "fp32", "", 0, 0)),
    std::runtime_error);
}

TEST_F(MigraphxCommonTest, ReplacedModelIsRebuiltNotRunStale)
{
  const auto dir = work_dir_ / "replaced";
  const auto model = dir / "model.onnx";
  placeModel("conv_network.onnx", model);
  {
    auto trt = makeAt(dir, model);
    ASSERT_TRUE(setupConv(*trt));
    EXPECT_LT(runConv(*trt, 8), 1e-5F);
  }
  const auto program = programIn(dir);
  ASSERT_FALSE(program.empty());

  // Other weights, same inputs and outputs: nothing but the model itself tells the programs apart.
  placeModel("conv_network_retrained.onnx", model);
  const auto retrained = readFloats(kDataDir / "conv_network_retrained_reference.bin");
  ASSERT_EQ(retrained.size(), reference_.size());
  std::filesystem::last_write_time(
    program, std::filesystem::last_write_time(program) - std::chrono::hours(1));
  auto before = std::filesystem::last_write_time(program);
  {
    auto trt = makeAt(dir, model);
    ASSERT_TRUE(setupConv(*trt));
    EXPECT_GT(std::filesystem::last_write_time(program), before) << "the program was not rebuilt";
    EXPECT_LT(runConv(*trt, 8, "image", kSize, &retrained), 1e-5F);
  }

  // Another input size, and the caller's NetworkIO updated to match.
  placeModel("conv_network_resized.onnx", model);
  const auto resized = readFloats(kDataDir / "conv_network_resized_reference.bin");
  ASSERT_EQ(resized.size(), reference_.size());
  std::filesystem::last_write_time(
    program, std::filesystem::last_write_time(program) - std::chrono::hours(1));
  before = std::filesystem::last_write_time(program);
  {
    auto trt = makeAt(dir, model);
    ASSERT_TRUE(setupConv(*trt, "image", 2 * kSize));
    EXPECT_GT(std::filesystem::last_write_time(program), before) << "the program was not rebuilt";
    expectDims(trt->getTensorShape("image"), {-1, kChannels, 2 * kSize, 2 * kSize});
    EXPECT_FALSE(trt->setInputShape("image", imageDims(8)));
    EXPECT_LT(runConv(*trt, 8, "image", 2 * kSize, &resized), 1e-5F);
  }
}

TEST_F(MigraphxCommonTest, RenamedInputIsRebuiltNotRefused)
{
  const auto dir = work_dir_ / "renamed";
  const auto model = dir / "model.onnx";
  placeModel("conv_network.onnx", model);
  {
    auto trt = makeAt(dir, model);
    ASSERT_TRUE(setupConv(*trt));
  }
  placeModel("conv_network_renamed.onnx", model);
  auto trt = makeAt(dir, model);
  EXPECT_STREQ(trt->getIOTensorName(0), "pixels");
  ASSERT_TRUE(setupConv(*trt, "pixels"));
  EXPECT_LT(runConv(*trt, 8, "pixels"), 1e-5F);
}

TEST_F(MigraphxCommonTest, CachedProgramThatDoesNotFitTheModelIsRebuilt)
{
  // A program compiled from another model under a key that claims this one, as a copied or
  // half-written cache could leave it: loading it fails, and that means rebuilding.
  const auto renamed_dir = work_dir_ / "other";
  placeModel("conv_network_renamed.onnx", renamed_dir / "model.onnx");
  {
    auto trt = makeAt(renamed_dir, renamed_dir / "model.onnx");
    ASSERT_TRUE(setupConv(*trt, "pixels"));
  }
  const auto dir = work_dir_ / "mismatch";
  placeModel("conv_network.onnx", dir / "model.onnx");
  {
    auto trt = makeAt(dir, dir / "model.onnx");
    ASSERT_TRUE(setupConv(*trt));
  }
  const auto program = programIn(dir);
  std::filesystem::copy_file(
    programIn(renamed_dir), program, std::filesystem::copy_options::overwrite_existing);
  std::filesystem::last_write_time(
    program, std::filesystem::last_write_time(program) - std::chrono::hours(1));
  const auto before = std::filesystem::last_write_time(program);

  auto trt = makeAt(dir, dir / "model.onnx");
  ASSERT_TRUE(setupConv(*trt));
  EXPECT_GT(std::filesystem::last_write_time(program), before) << "the program was not rebuilt";
  EXPECT_LT(runConv(*trt, 8), 1e-5F);
}

TEST_F(MigraphxCommonTest, SymbolicOutputDimsAreKnownOnlyWithStaticInputs)
{
  // Static input: every output dimension is known, as a TensorRT engine reports it.
  {
    auto trt = std::make_unique<TrtCommon>(TrtCommonConfig(
      (kDataDir / "conv_network_static_symbolic.onnx").string(), "fp32",
      enginePath("static_symbolic"), 1ULL << 26U));
    expectDims(trt->getOutputDims(0), {2, kClasses});
    const std::vector<NetworkIO> io{
      NetworkIO("image", imageDims(2)), NetworkIO("logits", {2, {2, kClasses}}),
      NetworkIO("features", {2, {2, kFeatures}})};
    ASSERT_TRUE(trt->setup(nullptr, std::make_unique<std::vector<NetworkIO>>(io)));
    expectDims(trt->getTensorShape("logits"), {2, kClasses});
    expectDims(trt->getTensorShape("features"), {2, kFeatures});
    EXPECT_LT(runConv(*trt, 2), 1e-5F);
  }
  // Dynamic batch: every dimension the model leaves symbolic is -1, also the class and feature
  // counts that TensorRT would report, so NetworkIO entries giving those lengths are refused.
  {
    const auto model = (kDataDir / "conv_network_dynamic_symbolic.onnx").string();
    auto refused = std::make_unique<TrtCommon>(
      TrtCommonConfig(model, "fp32", enginePath("dynamic_symbolic_refused"), 1ULL << 26U));
    expectDims(refused->getOutputDims(0), {-1, -1});
    expectDims(refused->getOutputDims(1), {-1, -1});
    EXPECT_FALSE(setupConv(*refused));

    auto trt = std::make_unique<TrtCommon>(
      TrtCommonConfig(model, "fp32", enginePath("dynamic_symbolic"), 1ULL << 26U));
    ASSERT_TRUE(trt->setup(
      std::make_unique<std::vector<ProfileDims>>(convProfile()),
      std::make_unique<std::vector<NetworkIO>>(std::vector<NetworkIO>{
        NetworkIO("image", {4, {-1, kChannels, kSize, kSize}}), NetworkIO("logits", {2, {-1, -1}}),
        NetworkIO("features", {2, {-1, -1}})})));
    expectDims(trt->getTensorShape("logits"), {-1, -1});
    expectDims(trt->getTensorShape("features"), {-1, -1});
    EXPECT_LT(runConv(*trt, 3), 1e-5F);
  }
}

TEST_F(MigraphxCommonTest, Fp16CanKeepNamedOperationsInFp32)
{
  constexpr int64_t kRows = 4;
  constexpr int64_t kColumns = 64;
  const auto reference = readFloats(kDataDir / "activations_reference.bin");
  ASSERT_EQ(reference.size(), 2U * kRows * kColumns);
  std::vector<float> x(kRows * kColumns);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = inputValue(i);
  }

  const auto max_error = [&](const std::vector<std::string> & excluded) {
    TrtCommonConfig config(
      (kDataDir / "activations.onnx").string(), "fp16", enginePath("activations"), 1ULL << 26U);
    config.fp16_excluded_ops = excluded;
    TrtCommon trt(config);
    EXPECT_TRUE(trt.setup());
    DeviceBuffer x_d(x.size());
    DeviceBuffer sigmoid_d(x.size());
    DeviceBuffer softmax_d(x.size());
    x_d.upload(x);
    EXPECT_TRUE(trt.setTensorAddress("x", x_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("sigmoid", sigmoid_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("softmax", softmax_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto outputs = sigmoid_d.download();
    const auto softmax = softmax_d.download();
    outputs.insert(outputs.end(), softmax.begin(), softmax.end());
    float error = 0.0F;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      error = std::max(error, std::fabs(outputs[i] - reference[i]));
    }
    return error;
  };

  // MIGraphX computes the matrix product in fp16 and, here, the activations after it in fp32.
  const auto quantized = max_error({"sigmoid", "softmax"});
  // The same engine path: the exclusions are part of what a cached program must match.
  const auto kept = max_error({"dot", "sigmoid", "softmax"});
  std::cout << "largest error: " << quantized << " with the product in fp16, " << kept
            << " with it in fp32" << std::endl;
  EXPECT_GT(quantized, 10.0F * kept) << "fp16 was not applied, or not undone";
  // fp32 rounding, with room for another GPU's exp; fp16 measures about 1.6e-4 here.
  EXPECT_LT(kept, 1e-5F);
}

TEST_F(MigraphxCommonTest, Fp16ExclusionsApplyToTheOptimizedProgram)
{
  // scaled_sigmoid.onnx divides by a constant, which MIGraphX makes a multiplication before it
  // converts operations to fp16: excluding the sigmoid still leaves that multiplication in fp16.
  constexpr int64_t kRows = 4;
  constexpr int64_t kColumns = 64;
  const auto reference = readFloats(kDataDir / "scaled_sigmoid_reference.bin");
  ASSERT_EQ(reference.size(), static_cast<std::size_t>(kRows * kColumns));
  std::vector<float> x(kRows * kColumns);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = inputValue(i);
  }

  const auto max_error = [&](const std::vector<std::string> & excluded) {
    TrtCommonConfig config(
      (kDataDir / "scaled_sigmoid.onnx").string(), "fp16", enginePath("scaled_sigmoid"),
      1ULL << 26U);
    config.fp16_excluded_ops = excluded;
    TrtCommon trt(config);
    EXPECT_TRUE(trt.setup());
    DeviceBuffer x_d(x.size());
    DeviceBuffer y_d(x.size());
    x_d.upload(x);
    EXPECT_TRUE(trt.setTensorAddress("x", x_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("y", y_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    const auto y = y_d.download();
    float error = 0.0F;
    for (std::size_t i = 0; i < y.size(); ++i) {
      error = std::max(error, std::fabs(y[i] - reference[i]));
    }
    return error;
  };

  const auto quantized = max_error({"sigmoid"});
  const auto kept = max_error({"mul", "sigmoid"});
  std::cout << "largest error: " << quantized << " with the multiplication in fp16, " << kept
            << " with it in fp32" << std::endl;
  EXPECT_GT(quantized, 10.0F * kept) << "the multiplication was not converted to fp16";
  EXPECT_LT(quantized, 1e-2F);
  EXPECT_LT(kept, 1e-5F);
}

TEST_F(MigraphxCommonTest, Fp16ExclusionsDoNotStopACachedProgramFromLoading)
{
  // setup() lists the operations to convert by reading the model, also when the program is cached.
  // If that fails (here the model is gone after construction), the cached program still loads and
  // runs; without a cached program, setup() fails.
  const auto dir = work_dir_ / "listing";
  const auto model = dir / "model.onnx";
  placeModel("conv_network.onnx", model);
  const auto make = [&]() {
    TrtCommonConfig config(model.string(), "fp16", (dir / "model.engine").string(), 1ULL << 26U);
    config.fp16_excluded_ops = {"relu"};
    return std::make_unique<TrtCommon>(config);
  };
  {
    auto trt = make();
    ASSERT_TRUE(setupConv(*trt));
  }
  auto cached = make();
  auto uncached = make();
  std::filesystem::remove(model);
  ASSERT_TRUE(setupConv(*cached));
  EXPECT_LT(runConv(*cached, 8), 1e-2F);
  std::filesystem::remove(programIn(dir));
  EXPECT_FALSE(setupConv(*uncached));
}

namespace
{
/// Protobuf encoding, enough to write an ONNX model.
std::string varint(uint64_t value)
{
  std::string bytes;
  do {
    auto byte = static_cast<uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0) {
      byte |= 0x80U;
    }
    bytes.push_back(static_cast<char>(byte));
  } while (value != 0);
  return bytes;
}

std::string field(const uint32_t number, const std::string & bytes)
{
  return varint((number << 3U) | 2U) + varint(bytes.size()) + bytes;
}

std::string field(const uint32_t number, const uint64_t value)
{
  return varint(number << 3U) + varint(value);
}

/// An ONNX ValueInfoProto for a float tensor.
std::string floatTensorInfo(const std::string & name, const std::vector<uint64_t> & dims)
{
  std::string shape;
  for (const auto d : dims) {
    shape += field(1, field(1, d));
  }
  return field(1, name) + field(2, field(1, field(1, uint64_t{1}) + field(2, shape)));
}

/// Element (k, n) of the weights of the model `writeLargeWeightModel` writes.
float largeWeight(const std::size_t k, const std::size_t n)
{
  return (static_cast<float>((k * 31 + n * 17) % 97) / 97.0F - 0.5F) / 64.0F;
}

/// y = Sigmoid(x [rows, k] x W [k, n]): a graph of two operations with k * n weights.
void writeLargeWeightModel(
  const std::filesystem::path & path, const std::size_t rows, const std::size_t k,
  const std::size_t n)
{
  std::string weights(k * n * sizeof(float), '\0');
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const auto value = largeWeight(i, j);
      std::memcpy(&weights[(i * n + j) * sizeof(float)], &value, sizeof(float));
    }
  }
  // TensorProto: dims 1, data_type 2 (1 = float), name 8, raw_data 9.
  const auto initializer = field(1, uint64_t{k}) + field(1, uint64_t{n}) + field(2, uint64_t{1}) +
                           field(8, std::string("w")) + field(9, weights);
  weights.clear();
  weights.shrink_to_fit();
  // NodeProto: input 1, output 2, op_type 4. GraphProto: node 1, name 2, initializer 5, input 11,
  // output 12. ModelProto: ir_version 1, graph 7, opset_import 8.
  const auto graph = field(
                       1, field(1, std::string("x")) + field(1, std::string("w")) +
                            field(2, std::string("product")) + field(4, std::string("MatMul"))) +
                     field(
                       1, field(1, std::string("product")) + field(2, std::string("y")) +
                            field(4, std::string("Sigmoid"))) +
                     field(2, std::string("large_weights")) + field(5, initializer) +
                     field(11, floatTensorInfo("x", {rows, k})) +
                     field(12, floatTensorInfo("y", {rows, n}));
  const auto model = field(1, uint64_t{9}) + field(7, graph) +
                     field(8, field(1, std::string()) + field(2, uint64_t{13}));
  std::ofstream(path, std::ios::binary)
    .write(model.data(), static_cast<std::streamsize>(model.size()));
}

/// The largest resident set size since the last `resetPeakMemory()`, in bytes.
std::size_t peakMemory()
{
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.rfind("VmHWM:", 0) == 0) {
      return std::stoull(line.substr(6)) * 1024;
    }
  }
  return 0;
}

std::size_t currentMemory()
{
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.rfind("VmRSS:", 0) == 0) {
      return std::stoull(line.substr(6)) * 1024;
    }
  }
  return 0;
}

bool resetPeakMemory()
{
  std::ofstream clear_refs("/proc/self/clear_refs");
  clear_refs << "5";
  clear_refs.close();
  return static_cast<bool>(clear_refs);
}
}  // namespace

TEST_F(MigraphxCommonTest, Fp16ExclusionsDoNotSerializeTheWeights)
{
  // 64 MiB of weights behind two operations. Listing the operations for fp16_excluded_ops must not
  // copy the weights into another form (in JSON they took about 40 times their size): building
  // with exclusions takes little more memory than building without. The peak is measured through
  // /proc/self/clear_refs, after a first build that pays MIGraphX's and HIP's one-time setup.
  if (!resetPeakMemory()) {
    GTEST_SKIP() << "/proc/self/clear_refs cannot be written, so the peak memory cannot be reset";
  }
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kInner = 4096;
  constexpr std::size_t kOuter = 4096;
  constexpr std::size_t kWeightBytes = kInner * kOuter * sizeof(float);
  const auto model = work_dir_ / "large_weights.onnx";
  writeLargeWeightModel(model, kRows, kInner, kOuter);

  std::vector<float> x(kRows * kInner);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = inputValue(i);
  }
  std::vector<double> reference(kRows * kOuter, 0.0);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t k = 0; k < kInner; ++k) {
      for (std::size_t n = 0; n < kOuter; ++n) {
        reference[r * kOuter + n] += static_cast<double>(x[r * kInner + k]) * largeWeight(k, n);
      }
    }
  }

  // Returns how much the resident memory grew at most while the model was read and compiled.
  const auto run = [&](const std::vector<std::string> & excluded, const std::string & tag) {
    TrtCommonConfig config(model.string(), "fp16", enginePath(tag), 1ULL << 26U);
    config.fp16_excluded_ops = excluded;
    EXPECT_TRUE(resetPeakMemory());
    const auto before = currentMemory();
    TrtCommon trt(config);
    EXPECT_TRUE(trt.setup());
    const auto growth = peakMemory() - before;
    std::cout << tag << ": resident memory grew by " << growth / (1U << 20U)
              << " MiB at most while the model was read and compiled" << std::endl;
    DeviceBuffer x_d(x.size());
    DeviceBuffer y_d(kRows * kOuter);
    x_d.upload(x);
    EXPECT_TRUE(trt.setTensorAddress("x", x_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("y", y_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    const auto y = y_d.download();
    float error = 0.0F;
    for (std::size_t i = 0; i < y.size(); ++i) {
      const auto expected = 1.0 / (1.0 + std::exp(-reference[i]));
      error = std::max(error, static_cast<float>(std::fabs(y[i] - expected)));
    }
    EXPECT_LT(error, 1e-2F) << tag;
    return growth;
  };
  (void)run({}, "large_weights_fp16_warm_up");
  const auto without = run({}, "large_weights_fp16");
  const auto with = run({"sigmoid"}, "large_weights_fp16_excluded");
  EXPECT_LT(with, without + 2 * kWeightBytes);
}

TEST_F(MigraphxCommonTest, OutputDimsFollowingADynamicOneAreDynamicAndNeverWrittenPast)
{
  // pool_network.onnx: x [1, 2, height, 8] -> halved [1, 2, height / 2, 4] and
  // sparse [1, 2, (height - 1) / 300 + 1, 8]. TensorRT reports both heights as -1. sparse has one
  // row for every height up to 300, including the profile's optimum here, 250.
  const nvinfer1::Dims min{4, {1, 2, 2, 8}};
  const nvinfer1::Dims opt{4, {1, 2, 250, 8}};
  const nvinfer1::Dims max{4, {1, 2, 900, 8}};
  const auto make = [&](const std::string & tag) {
    return std::make_unique<TrtCommon>(TrtCommonConfig(
      (kDataDir / "pool_network.onnx").string(), "fp32", enginePath(tag), 1ULL << 26U));
  };
  const auto setup = [&](TrtCommon & trt, const int64_t sparse_height) {
    return trt.setup(
      std::make_unique<std::vector<ProfileDims>>(
        std::vector<ProfileDims>{ProfileDims("x", min, opt, max)}),
      std::make_unique<std::vector<NetworkIO>>(std::vector<NetworkIO>{
        NetworkIO("x", {4, {1, 2, -1, 8}}), NetworkIO("halved", {4, {1, 2, -1, 4}}),
        NetworkIO("sparse", {4, {1, 2, sparse_height, 8}})}));
  };

  // Runs at `height` with output buffers of the given rows followed by a guard region, checks the
  // outputs against a CPU max pool, and returns whether the guard regions are untouched.
  static constexpr std::size_t kGuard = 1024;
  static constexpr float kSentinel = -12345.0F;
  const auto run = [&](TrtCommon & trt, const int64_t height, const int64_t sparse_rows) {
    std::vector<float> x(2 * height * 8);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = inputValue(i);
    }
    const auto halved_rows = height / 2;
    const auto halved_size = static_cast<std::size_t>(2 * halved_rows * 4);
    const auto sparse_size = static_cast<std::size_t>(2 * sparse_rows * 8);
    DeviceBuffer x_d(x.size());
    DeviceBuffer halved_d(halved_size + kGuard);
    DeviceBuffer sparse_d(sparse_size + kGuard);
    x_d.upload(x);
    halved_d.upload(std::vector<float>(halved_size + kGuard, kSentinel));
    sparse_d.upload(std::vector<float>(sparse_size + kGuard, kSentinel));
    EXPECT_TRUE(trt.setTensor("x", x_d.get(), {4, {1, 2, height, 8}}));
    EXPECT_TRUE(trt.setTensorAddress("halved", halved_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("sparse", sparse_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_)) << "height " << height;
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    const auto halved = halved_d.download();
    const auto sparse = sparse_d.download();
    const auto at = [&](int64_t c, int64_t row, int64_t column) {
      return x[(c * height + row) * 8 + column];
    };
    const auto true_sparse_rows = (height - 1) / 300 + 1;
    for (int64_t c = 0; c < 2; ++c) {
      for (int64_t row = 0; row < halved_rows; ++row) {
        for (int64_t column = 0; column < 4; ++column) {
          const auto expected = std::max(
            std::max(at(c, 2 * row, 2 * column), at(c, 2 * row, 2 * column + 1)),
            std::max(at(c, 2 * row + 1, 2 * column), at(c, 2 * row + 1, 2 * column + 1)));
          EXPECT_EQ(halved[(c * halved_rows + row) * 4 + column], expected) << "height " << height;
        }
      }
      for (int64_t row = 0; sparse_rows == true_sparse_rows && row < sparse_rows; ++row) {
        for (int64_t column = 0; column < 8; ++column) {
          EXPECT_EQ(sparse[(c * sparse_rows + row) * 8 + column], at(c, 300 * row, column))
            << "height " << height;
        }
      }
    }
    const auto untouched = [](const std::vector<float> & buffer, std::size_t from) {
      return std::all_of(
        buffer.begin() + static_cast<std::ptrdiff_t>(from), buffer.end(),
        [](float v) { return v == kSentinel; });
    };
    return untouched(halved, halved_size) && untouched(sparse, sparse_size);
  };

  // Before setup() the heights are -1, though sparse has one row at every small height.
  auto trt = make("pool");
  expectDims(trt->getOutputDims(0), {1, 2, -1, 4});
  expectDims(trt->getOutputDims(1), {1, 2, -1, 8});

  // A configuration that takes sparse to have one row, as the program compiled for the optimum
  // shows, is refused as with TensorRT: a buffer sized from it would be written past at 600.
  {
    auto one_row = make("pool_one_row");
    EXPECT_FALSE(setup(*one_row, 1));
    if (one_row->getTensorShape("sparse").d[2] == 1) {
      EXPECT_TRUE(run(*one_row, 600, 1))
        << "a run wrote past a buffer sized from the reported shape";
    }
  }

  ASSERT_TRUE(setup(*trt, -1));
  for (const int64_t height : std::vector<int64_t>{600, 250, 10, 301, 900, 2}) {
    EXPECT_TRUE(run(*trt, height, (height - 1) / 300 + 1)) << "height " << height;
    expectDims(trt->getTensorShape("halved"), {1, 2, -1, 4});
    expectDims(trt->getTensorShape("sparse"), {1, 2, -1, 8});
  }
}

namespace
{
// Profile of x [rows, 4] in the rows models: 1 to 64 rows, optimum 8.
std::unique_ptr<std::vector<ProfileDims>> rowsProfile()
{
  return std::make_unique<std::vector<ProfileDims>>(
    std::vector<ProfileDims>{ProfileDims("x", {2, {1, 4}}, {2, {8, 4}}, {2, {64, 4}})});
}
}  // namespace

TEST_F(MigraphxCommonTest, OutputsWhoseRankMigraphxCannotKeepAreRefused)
{
  // MIGraphX gives a scalar one element in one dimension, and does not always give an output the
  // rank the model computes, so the constructor refuses a model with an output that is a scalar,
  // that the model gives no shape, or that MIGraphX reads with another rank, and logs why.
  const std::vector<std::pair<std::string, std::string>> cases{
    {"scalar_sum", "it is a scalar (rank 0)"},
    {"scalar_sum_static", "it is a scalar (rank 0)"},
    {"scalar_sum_shapeless", "the model gives it no shape"},
    {"scalar_sum_shapeless_static", "the model gives it no shape"},
    {"shapeless_output", "the model gives it no shape"},
    {"squeezed", "the model gives it no shape"},
    {"unsqueezed_scalar", "the model gives it no shape"},
    {"unsqueezed_scalar_declared",
     "it has 2 dimensions in the model, but MIGraphX reads it as [1]"},
  };
  for (const auto & [name, reason] : cases) {
    ::testing::internal::CaptureStderr();
    EXPECT_THROW(
      TrtCommon(TrtCommonConfig(
        (kDataDir / (name + ".onnx")).string(), "fp32", enginePath(name), 1ULL << 26U)),
      std::runtime_error)
      << name;
    const auto log = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(
      log.find("Output y is not supported by the MIGraphX backend: " + reason), std::string::npos)
      << name << " logged:\n"
      << log;
  }
}

TEST_F(MigraphxCommonTest, OneElementOutputKeepsTheRankTheModelDeclares)
{
  // row_sums.onnx: y = the row sums of x [rows, 4], declared [rows]. With one row MIGraphX gives y
  // one element in one dimension, as it gives a scalar. y is reported as [-1] before and after
  // setup(), and a buffer sized from that, -1 taken as the number of rows, holds what a run
  // writes, also with one row.
  auto trt = std::make_unique<TrtCommon>(TrtCommonConfig(
    (kDataDir / "row_sums.onnx").string(), "fp32", enginePath("row_sums"), 1ULL << 26U));
  expectDims(trt->getOutputDims(0), {-1});
  ASSERT_TRUE(trt->setup(rowsProfile(), nullptr));
  expectDims(trt->getOutputDims(0), {-1});
  expectDims(trt->getTensorShape("y"), {-1});
  for (const int64_t rows : std::vector<int64_t>{64, 8, 1}) {
    const auto y = runRows(*trt, rows, {rows});
    ASSERT_EQ(y.size(), static_cast<std::size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) {
      float sum = 0.0F;
      for (int64_t c = 0; c < 4; ++c) {
        sum += inputValue(r * 4 + c);
      }
      EXPECT_NEAR(y[r], sum, 1e-5) << "rows " << rows << ", row " << r;
    }
  }
}

TEST_F(MigraphxCommonTest, CachedProgramWithOtherOutputDimsIsRebuilt)
{
  // A program whose output has another rank, or another length where one is reported, cached
  // under a key that claims this model: loading checks its outputs against the reported dimensions
  // and rebuilds.
  const auto replace_program = [&](
                                 const std::filesystem::path & dir, const std::string & other,
                                 const std::function<bool(TrtCommon &)> & setup) {
    const auto other_dir = dir.string() + "_other";
    placeModel(other, std::filesystem::path(other_dir) / "model.onnx");
    {
      auto trt = makeAt(other_dir, std::filesystem::path(other_dir) / "model.onnx");
      ASSERT_TRUE(setup(*trt)) << other;
    }
    const auto program = programIn(dir);
    std::filesystem::copy_file(
      programIn(other_dir), program, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::last_write_time(
      program, std::filesystem::last_write_time(program) - std::chrono::hours(1));
  };

  // y [rows, 4], reported [-1, 4], against a program giving y [rows * 4], which holds x itself, so
  // a run of that program would also fail the ReLU check.
  {
    const auto dir = work_dir_ / "rank";
    placeModel("rows_relu.onnx", dir / "model.onnx");
    const auto setup = [](TrtCommon & trt) { return trt.setup(rowsProfile(), nullptr); };
    {
      auto trt = makeAt(dir, dir / "model.onnx");
      ASSERT_TRUE(setup(*trt));
    }
    replace_program(dir, "rows_reshaped.onnx", setup);
    const auto before = std::filesystem::last_write_time(programIn(dir));
    auto trt = makeAt(dir, dir / "model.onnx");
    ASSERT_TRUE(setup(*trt));
    EXPECT_GT(std::filesystem::last_write_time(programIn(dir)), before)
      << "a program with another output rank was run";
    for (const int64_t rows : std::vector<int64_t>{64, 8, 3}) {
      expectRelu(runRows(*trt, rows, {rows, 4}), rows);
    }
  }
  // logits [batch, 2] against a program giving logits [batch, 3].
  {
    const auto dir = work_dir_ / "length";
    placeModel("conv_network.onnx", dir / "model.onnx");
    const auto setup = [](TrtCommon & trt) { return setupConv(trt); };
    {
      auto trt = makeAt(dir, dir / "model.onnx");
      ASSERT_TRUE(setup(*trt));
    }
    replace_program(dir, "conv_network_three_classes.onnx", [](TrtCommon & trt) {
      return trt.setup(
        std::make_unique<std::vector<ProfileDims>>(convProfile()),
        std::make_unique<std::vector<NetworkIO>>(std::vector<NetworkIO>{
          NetworkIO("image", imageDims(-1)), NetworkIO("logits", {2, {-1, 3}}),
          NetworkIO("features", {2, {-1, kFeatures}})}));
    });
    const auto before = std::filesystem::last_write_time(programIn(dir));
    auto trt = makeAt(dir, dir / "model.onnx");
    ASSERT_TRUE(setup(*trt));
    EXPECT_GT(std::filesystem::last_write_time(programIn(dir)), before)
      << "a program with another output length was run";
    EXPECT_LT(runConv(*trt, 8), 1e-5F);
  }
}

TEST_F(MigraphxCommonTest, ScalarInputIsSetWithNoDimensions)
{
  // scalar_scale.onnx: y = x * s for x [rows, 4] and a scalar s, which has rank 0.
  auto trt = std::make_unique<TrtCommon>(TrtCommonConfig(
    (kDataDir / "scalar_scale.onnx").string(), "fp32", enginePath("scalar_scale"), 1ULL << 26U));
  expectDims(trt->getTensorShape(1), {});
  ASSERT_TRUE(trt->setup(rowsProfile(), nullptr));
  expectDims(trt->getTensorShape("s"), {});
  for (const int64_t rows : std::vector<int64_t>{8, 3}) {
    const auto y = runRows(*trt, rows, {rows, 4}, {{"s", 2.5F}});
    ASSERT_EQ(y.size(), static_cast<std::size_t>(rows * 4));
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_FLOAT_EQ(y[i], inputValue(i) * 2.5F) << "rows " << rows << ", element " << i;
    }
  }
}

TEST_F(MigraphxCommonTest, MalformedModelIsRefused)
{
  // A model field 2^64 - 11 bytes long, and a graph input whose name is 2^64 - 1 bytes long.
  const std::vector<std::vector<unsigned char>> files{
    {0x12, 0xf5, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01},
    {0x3a, 0x0d, 0x5a, 0x0b, 0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}};
  for (std::size_t i = 0; i < files.size(); ++i) {
    const auto path = work_dir_ / ("malformed_" + std::to_string(i) + ".onnx");
    std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char *>(files[i].data()), files[i].size());  // NOLINT
    EXPECT_THROW(
      TrtCommon(TrtCommonConfig(path.string(), "fp32", enginePath("malformed"), 1ULL << 26U)),
      std::runtime_error)
      << "file " << i;
  }
}

TEST_F(MigraphxCommonTest, EmptyInputsRunWhenTheOutputsAllow)
{
  constexpr int64_t kColumns = 4;
  const auto rows_profile = []() {
    return std::make_unique<std::vector<ProfileDims>>(std::vector<ProfileDims>{
      ProfileDims("x", {2, {0, kColumns}}, {2, {4, kColumns}}, {2, {16, kColumns}})});
  };

  // MIGraphX compiles rows_scaled.onnx for no rows: the run computes the second output, b * 2,
  // and the empty input and output are given no memory.
  {
    ASSERT_TRUE(compilesFor("rows_scaled.onnx", "x", {0, kColumns}));
    TrtCommon trt(TrtCommonConfig(
      (kDataDir / "rows_scaled.onnx").string(), "fp32", enginePath("rows_scaled"), 1ULL << 26U));
    ASSERT_TRUE(trt.setup(rows_profile()));
    DeviceBuffer b(kColumns);
    DeviceBuffer scaled(kColumns);
    b.upload({1.0F, -2.0F, 3.5F, 0.25F});
    scaled.upload(std::vector<float>(kColumns, std::nanf("")));
    ASSERT_TRUE(trt.setTensorAddress("x", nullptr));
    ASSERT_TRUE(trt.setTensorAddress("b", b.get()));
    ASSERT_TRUE(trt.setTensorAddress("relu", nullptr));
    ASSERT_TRUE(trt.setTensorAddress("scaled", scaled.get()));
    ASSERT_TRUE(trt.setInputShape("x", {2, {0, kColumns}}));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    const std::vector<float> expected{2.0F, -4.0F, 7.0F, 0.5F};
    EXPECT_EQ(scaled.download(), expected);
  }

  // It does not compile the conv network for an empty batch, but every output is empty then, so
  // there is nothing to compute and nothing is run.
  {
    ASSERT_FALSE(compilesFor("conv_network.onnx", "image", {0, kChannels, kSize, kSize}));
    auto trt = makeConv("fp32", "empty_batch");
    ASSERT_TRUE(trt->setup(
      std::make_unique<std::vector<ProfileDims>>(
        std::vector<ProfileDims>{ProfileDims("image", imageDims(0), kImageOpt, kImageMax)})));
    DeviceBuffer sentinel(1);
    sentinel.upload({std::nanf("")});
    ASSERT_TRUE(trt->setTensor("image", sentinel.get(), imageDims(0)));
    ASSERT_TRUE(trt->setTensorAddress("logits", sentinel.get()));
    ASSERT_TRUE(trt->setTensorAddress("features", sentinel.get()));
    EXPECT_TRUE(trt->enqueueV3(stream_));
    EXPECT_TRUE(trt->enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    EXPECT_TRUE(std::isnan(sentinel.download().front()));
    // Batches the profile does not start at zero still refuse it.
    auto other = makeConv("fp32", "empty_batch_refused");
    ASSERT_TRUE(setupConv(*other));
    EXPECT_FALSE(other->setInputShape("image", imageDims(0)));
  }

  // The column sums of no rows are not empty, and MIGraphX cannot compute them: refused, not
  // skipped.
  {
    ASSERT_FALSE(compilesFor("rows.onnx", "x", {0, kColumns}));
    TrtCommon trt(
      TrtCommonConfig((kDataDir / "rows.onnx").string(), "fp32", enginePath("rows"), 1ULL << 26U));
    ASSERT_TRUE(trt.setup(rows_profile()));
    DeviceBuffer column_sum(kColumns);
    ASSERT_TRUE(trt.setTensorAddress("x", nullptr));
    ASSERT_TRUE(trt.setTensorAddress("relu", nullptr));
    ASSERT_TRUE(trt.setTensorAddress("column_sum", column_sum.get()));
    ASSERT_TRUE(trt.setInputShape("x", {2, {0, kColumns}}));
    EXPECT_FALSE(trt.enqueueV3(stream_));
  }
}

TEST_F(MigraphxCommonTest, PluginLibrariesAreLoadedBeforeTheModelIsRead)
{
  const std::string plugin{TEST_PLUGIN_PATH};
  const auto profiler = std::make_shared<autoware::tensorrt_common::Profiler>();

  // Loading the library registers its MIGraphX operator. MIGraphX's ONNX parser knows only its
  // own operators, so a model that calls the registered one is still refused.
  EXPECT_THROW(
    TrtCommon(
      TrtCommonConfig(
        (kDataDir / "custom_op_network.onnx").string(), "fp32", enginePath("plugin"), 1ULL << 26U),
      profiler, {plugin}),
    std::runtime_error);
  void * handle = dlopen(plugin.c_str(), RTLD_NOW | RTLD_NOLOAD);
  ASSERT_NE(handle, nullptr) << "plugin_paths did not load " << plugin;
  const auto registered = reinterpret_cast<int (*)()>(  // NOLINT
    dlsym(handle, "autoware_tensorrt_common_test_plugin_registered"));
  ASSERT_NE(registered, nullptr);
  EXPECT_EQ(registered(), 1);

  // The registered operator works in a program built through the MIGraphX API, where its output
  // buffer is an allocation passed as its last input: y = AutowareTestFillSeven(x, buffer) + x.
  migraphx::program program;
  auto main_module = program.get_main_module();
  const migraphx::shape shape{migraphx_shape_float_type, {16}};
  const auto x = main_module.add_parameter("x", shape);
  const auto sevens = main_module.add_instruction(
    migraphx::operation("AutowareTestFillSeven"), {x, main_module.add_allocation(shape)});
  main_module.add_return({main_module.add_instruction(migraphx::operation("add"), {sevens, x})});
  migraphx::compile_options options;
  options.set_offload_copy();
  program.compile(migraphx::target("gpu"), options);
  std::vector<float> values(16);
  std::iota(values.begin(), values.end(), 0.0F);
  migraphx::program_parameters parameters;
  parameters.add("x", migraphx::argument(shape, values.data()));
  const auto y = program.eval(parameters)[0].as_vector<float>();
  ASSERT_EQ(y.size(), values.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    EXPECT_EQ(y[i], values[i] + 7.0F) << "element " << i;
  }
  dlclose(handle);

  // A library that cannot be loaded is logged and skipped, as with TensorRT.
  TrtCommon trt(
    TrtCommonConfig(
      (kDataDir / "conv_network.onnx").string(), "fp32", enginePath("no_plugin"), 1ULL << 26U),
    profiler, {(work_dir_ / "missing.so").string()});
  ASSERT_TRUE(setupConv(trt));
  EXPECT_LT(runConv(trt, 8), 1e-5F);
}
