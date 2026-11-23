// Copyright 2023 TIER IV, Inc.
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

#include "autoware/tensorrt_common/logger.hpp"
#include "autoware/tensorrt_common/tensorrt_common.hpp"
#include "autoware/tensorrt_common/utils.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace autoware
{
namespace tensorrt_common
{
namespace
{
std::string joinNames(const std::set<std::string> & names)
{
  std::string joined;
  for (const auto & name : names) {
    joined += joined.empty() ? name : ", " + name;
  }
  return joined;
}

/// Erase every entry naming one of `names`.
template <typename IOEntries>
void eraseNamedIO(IOEntries & entries, const std::set<std::string> & names)
{
  const auto named = [&names](const auto & entry) { return names.count(entry.tensor_name) > 0; };
  entries.erase(std::remove_if(entries.begin(), entries.end(), named), entries.end());
}

/// The names a caller offered, classified by what the model declares. Nothing is acted on yet: an
/// offer carrying anything `missing` is refused before a single `skippable` entry is dropped.
struct OfferedIO
{
  std::set<std::string> names;      //!< Every distinct name the offer mentions.
  std::set<std::string> skippable;  //!< Undeclared, and offered as optional: safe to drop.
  std::set<std::string> missing;    //!< Undeclared, and not offered as optional: fatal.
};

/// Add one list of offered entries to `offer`, which accumulates over however many lists there are.
template <typename IOEntries>
void collectOffer(
  const IOEntries & entries, const std::unordered_set<std::string> & declared, OfferedIO & offer)
{
  for (const auto & entry : entries) {
    offer.names.emplace(entry.tensor_name);
    if (declared.count(entry.tensor_name) > 0) {
      continue;
    }
    auto & undeclared = entry.optional ? offer.skippable : offer.missing;
    undeclared.emplace(entry.tensor_name);
  }
}

/// Collect both offered lists, either of which may be absent, into one account of the offer.
OfferedIO collectOfferedIO(
  const std::vector<NetworkIO> * network_io, const std::vector<ProfileDims> * profile_dims,
  const std::unordered_set<std::string> & declared)
{
  OfferedIO offer;
  if (network_io) {
    collectOffer(*network_io, declared, offer);
  }
  if (profile_dims) {
    collectOffer(*profile_dims, declared, offer);
  }
  return offer;
}

/// Whether `network_io` names every tensor the model declares, logging any it does not.
bool offersEveryDeclaredTensor(
  const std::vector<NetworkIO> & network_io, const std::unordered_set<std::string> & declared,
  Logger & logger)
{
  std::set<std::string> unoffered(declared.begin(), declared.end());
  for (const auto & io : network_io) {
    unoffered.erase(io.tensor_name);
  }
  if (unoffered.empty()) {
    return true;
  }
  logger.log(
    nvinfer1::ILogger::Severity::kERROR, "Model declares IO tensor(s) that were not offered: [%s]",
    joinNames(unoffered).c_str());
  return false;
}

/// Log how much of the offered IO the model declares.
void logReconciledIO(
  const std::size_t offered, const std::set<std::string> & skipped, Logger & logger)
{
  if (skipped.empty()) {
    logger.log(
      nvinfer1::ILogger::Severity::kINFO, "Model declares all %zu offered IO tensors", offered);
    return;
  }
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    "Model declares %zu of %zu offered IO tensors; skipping the optional tensors it does not "
    "declare: [%s]",
    offered - skipped.size(), offered, joinNames(skipped).c_str());
}
}  // namespace

template <typename IOEntries>
void TrtCommon::resolveTensorNamesOrThrow(IOEntries & entries) const
{
  for (auto & entry : entries) {
    if (!entry.tensor_name.empty()) {
      continue;
    }
    const auto * name = getIOTensorName(entry.tensor_index);
    if (name == nullptr) {
      throw std::invalid_argument(
        "No IO tensor at index " + std::to_string(entry.tensor_index) + "; the model declares " +
        std::to_string(getNbIOTensors()) + " IO tensors.");
    }
    entry.tensor_name = name;
  }
}

void TrtCommon::dropSkippedIO(const std::set<std::string> & skipped)
{
  if (network_io_) {
    eraseNamedIO(*network_io_, skipped);
  }
  if (profile_dims_) {
    eraseNamedIO(*profile_dims_, skipped);
  }
  // Remembered so the by-name setters no-op for a tensor the caller may still bind
  // unconditionally.
  skipped_io_.insert(skipped.begin(), skipped.end());
}

bool TrtCommon::reconcileOfferedIO()
{
  const auto declared = getIOTensorNames();

  // Whether the caller offered a complete account of the model's IO, judged on the offer as it
  // arrived: dropping every entry of an all-optional offer below would otherwise read as a caller
  // that never made the promise.
  const auto accounts_for_all_io = network_io_ && !network_io_->empty();

  const auto offer = collectOfferedIO(network_io_.get(), profile_dims_.get(), declared);
  if (offer.names.empty()) {
    return true;
  }

  // A required tensor the model does not declare is a mismatch between the model and the caller
  // that no engine build can resolve, so refuse it before building one.
  if (!offer.missing.empty()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kERROR, "Model does not declare the required IO tensor(s): [%s]",
      joinNames(offer.missing).c_str());
    return false;
  }

  dropSkippedIO(offer.skippable);

  if (accounts_for_all_io && !offersEveryDeclaredTensor(*network_io_, declared, *logger_)) {
    return false;
  }

  logReconciledIO(offer.names.size(), offer.skippable, *logger_);
  return true;
}

bool TrtCommon::acceptOfferedIO(ProfileDimsPtr profile_dims, NetworkIOPtr network_io)
{
  if (profile_dims) {
    resolveTensorNamesOrThrow(*profile_dims);
  }
  if (network_io) {
    resolveTensorNamesOrThrow(*network_io);
  }

  profile_dims_ = std::move(profile_dims);
  network_io_ = std::move(network_io);

  return reconcileOfferedIO();
}

std::unordered_set<std::string> TrtCommon::getIOTensorNames() const
{
  const auto num_tensors = getNbIOTensors();
  std::unordered_set<std::string> names;
  names.reserve(static_cast<std::size_t>(num_tensors));
  for (int32_t index = 0; index < num_tensors; ++index) {
    const auto * name = getIOTensorName(index);
    if (name == nullptr) {
      throw std::runtime_error(
        "TensorRT reports " + std::to_string(num_tensors) + " IO tensors but cannot name index " +
        std::to_string(index) + ".");
    }
    names.emplace(name);
  }
  return names;
}

bool TrtCommon::validateNetworkIO()
{
  auto success = true;
  if (!network_io_ || network_io_->empty()) {
    logger_->log(
      nvinfer1::ILogger::Severity::kWARNING,
      "Network IO is empty, skipping validation. It might lead to undefined behavior");
    return success;
  }

  if (network_io_->size() != static_cast<size_t>(getNbIOTensors())) {
    std::string tensor_names = "[" + std::string(getIOTensorName(0));
    for (int32_t i = 1; i < getNbIOTensors(); ++i) {
      tensor_names += ", " + std::string(getIOTensorName(i));
    }
    tensor_names += "]";
    logger_->log(
      nvinfer1::ILogger::Severity::kWARNING,
      "Number of tensors in the engine (%d) does not match number of tensors in the config (%d). "
      "Tensors in the built engine: %s",
      getNbIOTensors(), network_io_->size(), tensor_names.c_str());
    success = false;
  }
  for (const auto & io : *network_io_) {
    NetworkIO tensor_from_engine{io.tensor_name, getTensorShape(io.tensor_name.c_str())};
    if (io != tensor_from_engine) {
      logger_->log(
        nvinfer1::ILogger::Severity::kERROR,
        "Invalid tensor. Current configuration: %s. Cached engine: %s", io.toString().c_str(),
        tensor_from_engine.toString().c_str());
      success = false;
    }
    if (io.dtype) {
      const auto actual_dtype = getTensorDataType(io.tensor_name.c_str());
      if (!actual_dtype || *actual_dtype != *io.dtype) {
        logger_->log(
          nvinfer1::ILogger::Severity::kERROR,
          "Tensor '%s' dtype mismatch. Current configuration: %s. Cached engine: %s",
          io.tensor_name.c_str(), dtype_name(*io.dtype),
          actual_dtype ? dtype_name(*actual_dtype) : "unknown");
        success = false;
      }
    }
  }

  return success;
}

bool TrtCommon::setTensorAddress(const int32_t index, void * data)
{
  auto const & name = getIOTensorName(index);
  return setTensorAddress(name, data);
}

bool TrtCommon::setTensorsAddresses(std::vector<void *> & tensors)
{
  bool success = true;
  for (std::size_t i = 0, e = tensors.size(); i < e; i++) {
    auto const name = getIOTensorName(i);
    success &= setTensorAddress(name, tensors.at(i));
  }
  return success;
}

bool TrtCommon::setTensorsAddresses(std::unordered_map<const char *, void *> & tensors)
{
  bool success = true;
  for (auto const & tensor : tensors) {
    success &= setTensorAddress(tensor.first, tensor.second);
  }
  return success;
}

bool TrtCommon::setInputShape(const int32_t index, const nvinfer1::Dims & dimensions)
{
  auto const & name = getIOTensorName(index);
  return setInputShape(name, dimensions);
}

bool TrtCommon::setInputsShapes(const std::vector<nvinfer1::Dims> & dimensions)
{
  bool success = true;
  for (std::size_t i = 0, e = dimensions.size(); i < e; i++) {
    success &= setInputShape(i, dimensions.at(i));
  }
  return success;
}

bool TrtCommon::setInputsShapes(const std::unordered_map<const char *, nvinfer1::Dims> & dimensions)
{
  bool success = true;
  for (auto const & dim : dimensions) {
    success &= setInputShape(dim.first, dim.second);
  }
  return success;
}

bool TrtCommon::setTensor(const int32_t index, void * data, nvinfer1::Dims dimensions)
{
  auto success = setTensorAddress(index, data);
  if (dimensions.nbDims > 0) {
    success &= setInputShape(index, dimensions);
  }
  return success;
}

bool TrtCommon::setTensor(const char * tensor_name, void * data, nvinfer1::Dims dimensions)
{
  auto success = setTensorAddress(tensor_name, data);
  if (dimensions.nbDims > 0) {
    success &= setInputShape(tensor_name, dimensions);
  }
  return success;
}

bool TrtCommon::setTensors(TensorsVec & tensors)
{
  bool success = true;
  for (std::size_t i = 0, e = tensors.size(); i < e; i++) {
    success &= setTensor(i, tensors.at(i).first, tensors.at(i).second);
  }
  return success;
}

bool TrtCommon::setTensors(TensorsMap & tensors)
{
  bool success = true;
  for (auto const & tensor : tensors) {
    success &= setTensor(tensor.first, tensor.second.first, tensor.second.second);
  }
  return success;
}

}  // namespace tensorrt_common
}  // namespace autoware
