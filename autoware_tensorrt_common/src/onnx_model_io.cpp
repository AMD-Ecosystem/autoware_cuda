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

#include "onnx_model_io.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// The model file is a protobuf ModelProto (onnx/onnx.proto). Decoding the few fields needed here
// directly keeps protobuf and the ONNX schema out of this package's dependencies.

namespace autoware
{
namespace tensorrt_common
{
namespace
{
// Protobuf wire types.
constexpr uint32_t kVarint = 0;
constexpr uint32_t kFixed64 = 1;
constexpr uint32_t kLengthDelimited = 2;
constexpr uint32_t kFixed32 = 5;

// Field numbers from onnx.proto.
constexpr uint32_t kModelGraph = 7;
constexpr uint32_t kGraphInitializer = 5;
constexpr uint32_t kGraphInput = 11;
constexpr uint32_t kGraphOutput = 12;
constexpr uint32_t kTensorName = 8;
constexpr uint32_t kValueInfoName = 1;
constexpr uint32_t kValueInfoType = 2;
constexpr uint32_t kTypeTensorType = 1;
constexpr uint32_t kTensorTypeElemType = 1;
constexpr uint32_t kTensorTypeShape = 2;
constexpr uint32_t kShapeDim = 1;
constexpr uint32_t kDimValue = 1;

/// Reads protobuf fields from a byte range of a stream, seeking over the ones it skips.
class WireReader
{
public:
  WireReader(std::istream & in, std::streamoff end) : in_(in), end_(end) {}

  /// Read the next field's key, or return false at the end of the range or on an error.
  bool next(uint32_t & field, uint32_t & wire_type)
  {
    if (!ok_ || in_.tellg() >= end_) {
      return false;
    }
    uint64_t key = 0;
    if (!varint(key)) {
      return false;
    }
    field = static_cast<uint32_t>(key >> 3U);
    wire_type = static_cast<uint32_t>(key & 7U);
    return true;
  }

  bool varint(uint64_t & value)
  {
    value = 0;
    for (uint32_t shift = 0; shift < 64; shift += 7) {
      const auto byte = in_.get();
      if (byte == std::char_traits<char>::eof()) {
        return fail();
      }
      value |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        return true;
      }
    }
    return fail();
  }

  /// Read a length prefix and return where the length-delimited field ends.
  bool length(std::streamoff & field_end)
  {
    uint64_t size = 0;
    if (!varint(size)) {
      return false;
    }
    const std::streamoff position = in_.tellg();
    if (position < 0 || position > end_ || size > static_cast<uint64_t>(end_ - position)) {
      return fail();
    }
    field_end = position + static_cast<std::streamoff>(size);
    return true;
  }

  bool string(std::string & value)
  {
    std::streamoff field_end = 0;
    if (!length(field_end)) {
      return false;
    }
    value.resize(static_cast<std::size_t>(field_end - in_.tellg()));
    in_.read(value.data(), static_cast<std::streamsize>(value.size()));
    return static_cast<bool>(in_) || fail();
  }

  bool skip(uint32_t wire_type)
  {
    uint64_t ignored = 0;
    std::streamoff field_end = 0;
    switch (wire_type) {
      case kVarint:
        return varint(ignored);
      case kFixed64:
        return seek(in_.tellg() + std::streamoff{8});
      case kLengthDelimited:
        return length(field_end) && seek(field_end);
      case kFixed32:
        return seek(in_.tellg() + std::streamoff{4});
      default:
        return fail();
    }
  }

  bool seek(std::streamoff position)
  {
    if (position > end_) {
      return fail();
    }
    in_.seekg(position);
    return static_cast<bool>(in_) || fail();
  }

  [[nodiscard]] bool ok() const { return ok_; }

  std::istream & stream() { return in_; }

private:
  bool fail()
  {
    ok_ = false;
    return false;
  }

  std::istream & in_;
  std::streamoff end_;
  bool ok_{true};
};

/// Read a length-delimited sub-message with `read(field_reader)`, then continue after it.
template <typename Read>
bool readMessage(WireReader & reader, Read read)
{
  std::streamoff field_end = 0;
  if (!reader.length(field_end)) {
    return false;
  }
  WireReader field_reader(reader.stream(), field_end);
  if (!read(field_reader) || !field_reader.ok()) {
    return false;
  }
  return reader.seek(field_end);
}

bool readDim(WireReader & reader, std::vector<int64_t> & dims)
{
  int64_t dim = -1;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    uint64_t value = 0;
    if (field == kDimValue && wire_type == kVarint) {
      if (!reader.varint(value)) {
        return false;
      }
      dim = static_cast<int64_t>(value);
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  // A symbolic dimension (dim_param) or a non-positive value is dynamic.
  dims.push_back(dim > 0 ? dim : -1);
  return true;
}

bool readShape(WireReader & reader, OnnxTensor & tensor)
{
  std::vector<int64_t> dims;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    if (field == kShapeDim && wire_type == kLengthDelimited) {
      if (!readMessage(reader, [&dims](WireReader & r) { return readDim(r, dims); })) {
        return false;
      }
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  tensor.dims = std::move(dims);
  return true;
}

bool readTensorType(WireReader & reader, OnnxTensor & tensor)
{
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    uint64_t value = 0;
    if (field == kTensorTypeElemType && wire_type == kVarint) {
      if (!reader.varint(value)) {
        return false;
      }
      tensor.elem_type = static_cast<int32_t>(value);
    } else if (field == kTensorTypeShape && wire_type == kLengthDelimited) {
      if (!readMessage(reader, [&tensor](WireReader & r) { return readShape(r, tensor); })) {
        return false;
      }
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  return true;
}

bool readType(WireReader & reader, OnnxTensor & tensor)
{
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    if (field == kTypeTensorType && wire_type == kLengthDelimited) {
      if (!readMessage(reader, [&tensor](WireReader & r) { return readTensorType(r, tensor); })) {
        return false;
      }
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  return true;
}

bool readValueInfo(WireReader & reader, std::vector<OnnxTensor> & tensors)
{
  OnnxTensor tensor;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    if (field == kValueInfoName && wire_type == kLengthDelimited) {
      if (!reader.string(tensor.name)) {
        return false;
      }
    } else if (field == kValueInfoType && wire_type == kLengthDelimited) {
      if (!readMessage(reader, [&tensor](WireReader & r) { return readType(r, tensor); })) {
        return false;
      }
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  tensors.push_back(std::move(tensor));
  return true;
}

bool readInitializerName(WireReader & reader, std::unordered_set<std::string> & names)
{
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    if (field == kTensorName && wire_type == kLengthDelimited) {
      std::string name;
      if (!reader.string(name)) {
        return false;
      }
      names.insert(std::move(name));
    } else if (!reader.skip(wire_type)) {
      return false;
    }
  }
  return true;
}

bool readGraph(WireReader & reader, OnnxModelIO & io)
{
  std::unordered_set<std::string> initializers;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    if (field == kGraphInput && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&io](WireReader & r) { return readValueInfo(r, io.inputs); });
    } else if (field == kGraphOutput && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&io](WireReader & r) { return readValueInfo(r, io.outputs); });
    } else if (field == kGraphInitializer && wire_type == kLengthDelimited) {
      read = readMessage(
        reader, [&initializers](WireReader & r) { return readInitializerName(r, initializers); });
    } else {
      read = reader.skip(wire_type);
    }
    if (!read) {
      return false;
    }
  }
  const auto is_initializer = [&initializers](const OnnxTensor & tensor) {
    return initializers.count(tensor.name) > 0;
  };
  io.inputs.erase(
    std::remove_if(io.inputs.begin(), io.inputs.end(), is_initializer), io.inputs.end());
  return reader.ok();
}
}  // namespace

std::optional<OnnxModelIO> readOnnxModelIO(const std::string & path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return std::nullopt;
  }
  WireReader reader(in, in.tellg());
  in.seekg(0);

  OnnxModelIO io;
  bool has_graph = false;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    if (field == kModelGraph && wire_type == kLengthDelimited) {
      has_graph = true;
      read = readMessage(reader, [&io](WireReader & r) { return readGraph(r, io); });
    } else {
      read = reader.skip(wire_type);
    }
    if (!read) {
      return std::nullopt;
    }
  }
  if (!reader.ok() || !has_graph) {
    return std::nullopt;
  }
  return io;
}
}  // namespace tensorrt_common
}  // namespace autoware
