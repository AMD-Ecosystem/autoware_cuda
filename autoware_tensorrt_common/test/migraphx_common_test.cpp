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
// IO order, dynamic shapes, precisions, and the compiled program cache.

#include <autoware/cuda_utils/cuda_gtest_utils.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <autoware/tensorrt_common/tensorrt_conv_calib.hpp>
#include <autoware/tensorrt_common/utils.hpp>

#include <NvInfer.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
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

nvinfer1::Dims imageDims(const int64_t batch)
{
  return nvinfer1::Dims{4, {batch, kChannels, kSize, kSize}};
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

    work_dir_ = std::filesystem::temp_directory_path() /
                ("autoware_tensorrt_common_migraphx_test_" +
                 std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(work_dir_);
    std::filesystem::create_directories(work_dir_);
    ASSERT_HIP_SUCCESS(hipStreamCreate(&stream_));

    std::ifstream file(kDataDir / "conv_network_reference.bin", std::ios::binary);
    reference_.resize(kReferenceBatch * (kClasses + kFeatures));
    file.read(
      reinterpret_cast<char *>(reference_.data()),  // NOLINT
      static_cast<std::streamsize>(reference_.size() * sizeof(float)));
    ASSERT_TRUE(file) << "missing conv_network_reference.bin";
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

  static std::vector<NetworkIO> convNetworkIO()
  {
    return {
      NetworkIO("image", {4, {-1, kChannels, kSize, kSize}}),
      NetworkIO("logits", {2, {-1, kClasses}}), NetworkIO("features", {2, {-1, kFeatures}})};
  }

  static bool setupConv(TrtCommon & trt)
  {
    return trt.setup(
      std::make_unique<std::vector<ProfileDims>>(convProfile()),
      std::make_unique<std::vector<NetworkIO>>(convNetworkIO()));
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

  /// Run the conv network on `batch` samples and return the largest error against the
  /// reference, with the outputs read from the buffers the caller bound.
  float runConv(TrtCommon & trt, const int64_t batch)
  {
    std::vector<float> image(batch * kChannels * kSize * kSize);
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

    EXPECT_TRUE(trt.setTensor("image", image_d.get(), imageDims(batch)));
    EXPECT_TRUE(trt.setTensorAddress("logits", logits_d.get()));
    EXPECT_TRUE(trt.setTensorAddress("features", features_d.get()));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    const auto logits = logits_d.download();
    const auto features = features_d.download();
    const float * reference_logits = reference_.data();
    const float * reference_features = reference_.data() + kReferenceBatch * kClasses;
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
  EXPECT_LT(runConv(*trt, 8), 1e-2F);
  ASSERT_EQ(programFiles().size(), 1U);
  EXPECT_NE(programFiles().front().filename().string().find(".fp16.mxr"), std::string::npos);
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
