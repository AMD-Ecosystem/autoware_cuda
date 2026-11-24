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

#include <NvInfer.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
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

TEST_F(MigraphxCommonTest, SymbolicOutputDimsAreReportedOnceKnown)
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
  // Dynamic batch: the dimensions that follow the batch stay -1, the others are known.
  {
    auto trt = std::make_unique<TrtCommon>(TrtCommonConfig(
      (kDataDir / "conv_network_dynamic_symbolic.onnx").string(), "fp32",
      enginePath("dynamic_symbolic"), 1ULL << 26U));
    ASSERT_TRUE(setupConv(*trt));
    expectDims(trt->getTensorShape("logits"), {-1, kClasses});
    expectDims(trt->getTensorShape("features"), {-1, kFeatures});
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
  EXPECT_GT(quantized, 10.0F * kept) << "fp16 was not applied, or not undone";
  EXPECT_LT(kept, 1e-6F);
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

  // MIGraphX compiles this model for no rows: an empty run that touches nothing.
  {
    TrtCommon trt(TrtCommonConfig(
      (kDataDir / "rows_relu.onnx").string(), "fp32", enginePath("rows_relu"), 1ULL << 26U));
    ASSERT_TRUE(trt.setup(rows_profile()));
    ASSERT_TRUE(trt.setTensorAddress("x", nullptr));
    ASSERT_TRUE(trt.setTensorAddress("relu", nullptr));
    ASSERT_TRUE(trt.setInputShape("x", {2, {0, kColumns}}));
    EXPECT_TRUE(trt.enqueueV3(stream_));
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  }

  // It does not compile the conv network for an empty batch, but every output is empty then, so
  // there is nothing to compute.
  {
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
  dlclose(handle);

  // A library that cannot be loaded is logged and skipped, as with TensorRT.
  TrtCommon trt(
    TrtCommonConfig(
      (kDataDir / "conv_network.onnx").string(), "fp32", enginePath("no_plugin"), 1ULL << 26U),
    profiler, {(work_dir_ / "missing.so").string()});
  ASSERT_TRUE(setupConv(trt));
  EXPECT_LT(runConv(trt, 8), 1e-5F);
}
