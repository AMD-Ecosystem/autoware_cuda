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
#include <unordered_map>
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
constexpr uint32_t kGraphNode = 1;
constexpr uint32_t kGraphInitializer = 5;
constexpr uint32_t kGraphInput = 11;
constexpr uint32_t kGraphOutput = 12;
constexpr uint32_t kGraphValueInfo = 13;
constexpr uint32_t kNodeInput = 1;
constexpr uint32_t kNodeOutput = 2;
constexpr uint32_t kNodeOpType = 4;
constexpr uint32_t kNodeAttribute = 5;
constexpr uint32_t kNodeDomain = 7;
constexpr uint32_t kAttributeName = 1;
constexpr uint32_t kAttributeInt = 3;
constexpr uint32_t kAttributeTensor = 5;
constexpr uint32_t kAttributeInts = 8;
constexpr uint32_t kTensorDims = 1;
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

  [[nodiscard]] std::streamoff end() const { return end_; }

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

/// Read an int64 field, packed or not, appending its values.
bool readInt64s(WireReader & reader, uint32_t wire_type, std::vector<int64_t> & values)
{
  uint64_t value = 0;
  if (wire_type == kVarint) {
    if (!reader.varint(value)) {
      return false;
    }
    values.push_back(static_cast<int64_t>(value));
    return true;
  }
  if (wire_type != kLengthDelimited) {
    return reader.skip(wire_type);
  }
  return readMessage(reader, [&values](WireReader & r) {
    uint64_t element = 0;
    while (r.stream().tellg() < r.end()) {
      if (!r.varint(element)) {
        return false;
      }
      values.push_back(static_cast<int64_t>(element));
    }
    return r.stream().tellg() == r.end();
  });
}

/// A tensor's name and dimensions, its data skipped.
bool readTensor(WireReader & reader, std::string & name, std::vector<int64_t> & dims)
{
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    if (field == kTensorName && wire_type == kLengthDelimited) {
      read = reader.string(name);
    } else if (field == kTensorDims) {
      read = readInt64s(reader, wire_type, dims);
    } else {
      read = reader.skip(wire_type);
    }
    if (!read) {
      return false;
    }
  }
  return true;
}

/// A node attribute: an integer, integers, or the dimensions of a tensor.
struct Attribute
{
  int64_t value{0};
  std::vector<int64_t> values;
  std::optional<std::vector<int64_t>> tensor_dims;
};

struct Node
{
  std::string op_type;
  std::string domain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, Attribute> attributes;
};

bool readAttribute(WireReader & reader, Node & node)
{
  std::string name;
  Attribute attribute;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    uint64_t value = 0;
    if (field == kAttributeName && wire_type == kLengthDelimited) {
      read = reader.string(name);
    } else if (field == kAttributeInt && wire_type == kVarint) {
      read = reader.varint(value);
      attribute.value = static_cast<int64_t>(value);
    } else if (field == kAttributeInts) {
      read = readInt64s(reader, wire_type, attribute.values);
    } else if (field == kAttributeTensor && wire_type == kLengthDelimited) {
      std::string ignored;
      std::vector<int64_t> dims;
      read = readMessage(
        reader, [&ignored, &dims](WireReader & r) { return readTensor(r, ignored, dims); });
      attribute.tensor_dims = std::move(dims);
    } else {
      read = reader.skip(wire_type);
    }
    if (!read) {
      return false;
    }
  }
  node.attributes[name] = std::move(attribute);
  return true;
}

bool readNode(WireReader & reader, std::vector<Node> & nodes)
{
  Node node;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    if (field == kNodeInput && wire_type == kLengthDelimited) {
      read = reader.string(node.inputs.emplace_back());
    } else if (field == kNodeOutput && wire_type == kLengthDelimited) {
      read = reader.string(node.outputs.emplace_back());
    } else if (field == kNodeOpType && wire_type == kLengthDelimited) {
      read = reader.string(node.op_type);
    } else if (field == kNodeDomain && wire_type == kLengthDelimited) {
      read = reader.string(node.domain);
    } else if (field == kNodeAttribute && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&node](WireReader & r) { return readAttribute(r, node); });
    } else {
      read = reader.skip(wire_type);
    }
    if (!read) {
      return false;
    }
  }
  nodes.push_back(std::move(node));
  return true;
}

/**
 * Ranks of a graph's tensors, inferred from the ONNX operator definitions. Only the rank is
 * inferred, so the operators covered are those whose output rank follows from the ranks of their
 * inputs, their attributes and the shapes of constant inputs (an axes list, a target shape). For
 * any other operator, or an operator outside the default domain, the rank is not known.
 */
class RankInference
{
public:
  RankInference(
    const std::vector<Node> & nodes, std::unordered_map<std::string, std::vector<int64_t>> shapes)
  : shapes_(std::move(shapes))
  {
    for (const auto & node : nodes) {
      for (std::size_t k = 0; k < node.outputs.size(); ++k) {
        producers_.emplace(node.outputs[k], std::make_pair(&node, k));
      }
    }
  }

  std::optional<std::size_t> rank(const std::string & name)
  {
    if (const auto shape = shapes_.find(name); shape != shapes_.end()) {
      return shape->second.size();
    }
    if (const auto known = ranks_.find(name); known != ranks_.end()) {
      return known->second;
    }
    const auto producer = producers_.find(name);
    if (producer == producers_.end() || !visiting_.insert(name).second) {
      return std::nullopt;
    }
    const auto result = nodeRank(*producer->second.first, producer->second.second);
    visiting_.erase(name);
    ranks_[name] = result;
    return result;
  }

private:
  /// The number of elements of a constant one-dimensional input such as an axes list.
  std::optional<std::size_t> length(const std::string & name)
  {
    if (
      const auto shape = shapes_.find(name);
      shape != shapes_.end() && shape->second.size() == 1 && shape->second[0] >= 0) {
      return static_cast<std::size_t>(shape->second[0]);
    }
    const auto producer = producers_.find(name);
    if (producer != producers_.end() && producer->second.first->op_type == "Constant") {
      const auto & attributes = producer->second.first->attributes;
      if (const auto ints = attributes.find("value_ints"); ints != attributes.end()) {
        return ints->second.values.size();
      }
      if (
        const auto value = attributes.find("value");
        value != attributes.end() && value->second.tensor_dims &&
        value->second.tensor_dims->size() == 1 && (*value->second.tensor_dims)[0] >= 0) {
        return static_cast<std::size_t>((*value->second.tensor_dims)[0]);
      }
    }
    return std::nullopt;
  }

  /// The number of axes a node names in its `axes` attribute or its input `axes_input`, nullopt
  /// when it names none; `known` is false when the number cannot be told.
  std::optional<std::size_t> axes(
    const Node & node, std::size_t axes_input, std::optional<std::size_t> input_rank, bool & known)
  {
    known = true;
    if (const auto attribute = node.attributes.find("axes"); attribute != node.attributes.end()) {
      std::unordered_set<int64_t> distinct;
      for (auto axis : attribute->second.values) {
        if (input_rank && axis < 0) {
          axis += static_cast<int64_t>(*input_rank);
        }
        distinct.insert(axis);
      }
      return distinct.size();
    }
    if (axes_input < node.inputs.size() && !node.inputs[axes_input].empty()) {
      const auto count = length(node.inputs[axes_input]);
      known = count.has_value();
      return count.value_or(0);
    }
    return std::nullopt;
  }

  int64_t intAttribute(const Node & node, const std::string & name, int64_t fallback) const
  {
    const auto attribute = node.attributes.find(name);
    return attribute == node.attributes.end() ? fallback : attribute->second.value;
  }

  static std::optional<std::size_t> minus(
    const std::optional<std::size_t> rank, const std::size_t count)
  {
    if (!rank || count > *rank) {
      return std::nullopt;
    }
    return *rank - count;
  }

  std::optional<std::size_t> nodeRank(const Node & node, const std::size_t output)
  {
    static const std::unordered_set<std::string> kSameRank{
      "Abs",
      "Acos",
      "Acosh",
      "AveragePool",
      "BatchNormalization",
      "BitwiseNot",
      "Cast",
      "CastLike",
      "Ceil",
      "Celu",
      "Clip",
      "Concat",
      "Conv",
      "ConvInteger",
      "ConvTranspose",
      "Cos",
      "Cosh",
      "CumSum",
      "DepthToSpace",
      "DequantizeLinear",
      "Dropout",
      "Elu",
      "Erf",
      "Exp",
      "Floor",
      "GatherElements",
      "Gelu",
      "GlobalAveragePool",
      "GlobalLpPool",
      "GlobalMaxPool",
      "GridSample",
      "GroupNormalization",
      "HardSigmoid",
      "HardSwish",
      "Hardmax",
      "Identity",
      "InstanceNormalization",
      "IsInf",
      "IsNaN",
      "LRN",
      "LayerNormalization",
      "LeakyRelu",
      "Log",
      "LogSoftmax",
      "LpNormalization",
      "LpPool",
      "MaxPool",
      "MeanVarianceNormalization",
      "Mish",
      "Neg",
      "Not",
      "Pad",
      "QuantizeLinear",
      "Reciprocal",
      "Relu",
      "Resize",
      "ReverseSequence",
      "Round",
      "ScatterElements",
      "ScatterND",
      "Selu",
      "Shrink",
      "Sigmoid",
      "Sign",
      "Sin",
      "Sinh",
      "Slice",
      "Softmax",
      "Softplus",
      "Softsign",
      "SpaceToDepth",
      "Split",
      "Sqrt",
      "Tan",
      "Tanh",
      "ThresholdedRelu",
      "Tile",
      "TopK",
      "Transpose",
      "Trilu",
      "Upsample"};
    // Of these, all outputs have the rank of the first input; of the others, only the first does.
    static const std::unordered_set<std::string> kSameRankEveryOutput{"Dropout", "Split", "TopK"};
    static const std::unordered_set<std::string> kBroadcast{
      "Add",        "And",         "BitShift", "BitwiseAnd", "BitwiseOr",
      "BitwiseXor", "Div",         "Equal",    "Greater",    "GreaterOrEqual",
      "Less",       "LessOrEqual", "Max",      "Mean",       "Min",
      "Mod",        "Mul",         "Or",       "PRelu",      "Pow",
      "Sub",        "Sum",         "Where",    "Xor"};
    static const std::unordered_set<std::string> kReduce{
      "ReduceL1",   "ReduceL2",  "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
      "ReduceMean", "ReduceMin", "ReduceProd",   "ReduceSum",       "ReduceSumSquare"};

    if (!node.domain.empty() && node.domain != "ai.onnx") {
      return std::nullopt;
    }
    const auto & op = node.op_type;
    const auto input = [&](std::size_t i) -> std::optional<std::size_t> {
      if (i >= node.inputs.size() || node.inputs[i].empty()) {
        return std::nullopt;
      }
      return rank(node.inputs[i]);
    };

    if (kSameRank.count(op) > 0) {
      if (output > 0 && kSameRankEveryOutput.count(op) == 0) {
        return std::nullopt;
      }
      return input(0);
    }
    if (output > 0) {
      return std::nullopt;
    }
    if (kBroadcast.count(op) > 0) {
      std::size_t result = 0;
      for (std::size_t i = 0; i < node.inputs.size(); ++i) {
        const auto r = input(i);
        if (!r) {
          return std::nullopt;
        }
        result = std::max(result, *r);
      }
      return result;
    }
    if (kReduce.count(op) > 0 || op == "ArgMax" || op == "ArgMin") {
      const auto r = input(0);
      if (intAttribute(node, "keepdims", 1) != 0) {
        return r;
      }
      if (op == "ArgMax" || op == "ArgMin") {
        return minus(r, 1);
      }
      bool known = true;
      const auto count = axes(node, 1, r, known);
      if (!known) {
        return std::nullopt;
      }
      if (!count || *count == 0) {
        return intAttribute(node, "noop_with_empty_axes", 0) != 0 ? r
                                                                  : std::optional<std::size_t>(0);
      }
      return minus(r, *count);
    }
    if (op == "Squeeze" || op == "Unsqueeze") {
      const auto r = input(0);
      bool known = true;
      const auto count = axes(node, 1, op == "Squeeze" ? r : std::nullopt, known);
      // Squeeze without axes drops the dimensions of length 1, which the rank does not tell.
      if (!known || !count || !r) {
        return std::nullopt;
      }
      return op == "Squeeze" ? minus(r, *count) : std::optional<std::size_t>(*r + *count);
    }
    if (op == "Reshape" || op == "ConstantOfShape") {
      const std::size_t shape = op == "Reshape" ? 1 : 0;
      return shape < node.inputs.size() ? length(node.inputs[shape]) : std::nullopt;
    }
    if (op == "Expand") {
      const auto r = input(0);
      const auto target = node.inputs.size() > 1 ? length(node.inputs[1]) : std::nullopt;
      if (!r || !target) {
        return std::nullopt;
      }
      return std::max(*r, *target);
    }
    if (op == "Flatten" || op == "Gemm" || op == "NonZero" || op == "NonMaxSuppression") {
      return 2;
    }
    if (op == "MatMul" || op == "MatMulInteger") {
      const auto a = input(0);
      const auto b = input(1);
      if (!a || !b || *a == 0 || *b == 0) {
        return std::nullopt;
      }
      if (*a == 1 && *b == 1) {
        return 0;
      }
      if (*a == 1 || *b == 1) {
        return std::max(*a, *b) - 1;
      }
      return std::max(*a, *b);
    }
    if (op == "Gather") {
      const auto data = input(0);
      const auto indices = input(1);
      if (!data || !indices || *data == 0) {
        return std::nullopt;
      }
      return *data + *indices - 1;
    }
    if (op == "Shape" || op == "Range") {
      return 1;
    }
    if (op == "Size") {
      return 0;
    }
    if (op == "Constant") {
      for (const auto & [name, attribute] : node.attributes) {
        if (name == "value" && attribute.tensor_dims) {
          return attribute.tensor_dims->size();
        }
        if (name == "value_float" || name == "value_int" || name == "value_string") {
          return 0;
        }
        if (name == "value_floats" || name == "value_ints" || name == "value_strings") {
          return 1;
        }
      }
    }
    return std::nullopt;
  }

  //!< Tensors whose dimensions the model states: graph inputs, initializers and value_info.
  std::unordered_map<std::string, std::vector<int64_t>> shapes_;
  std::unordered_map<std::string, std::pair<const Node *, std::size_t>> producers_;
  std::unordered_map<std::string, std::optional<std::size_t>> ranks_;
  std::unordered_set<std::string> visiting_;
};

bool readGraph(WireReader & reader, OnnxModelIO & io)
{
  std::unordered_set<std::string> initializers;
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  std::vector<OnnxTensor> value_info;
  std::vector<Node> nodes;
  uint32_t field = 0;
  uint32_t wire_type = 0;
  while (reader.next(field, wire_type)) {
    bool read = true;
    if (field == kGraphInput && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&io](WireReader & r) { return readValueInfo(r, io.inputs); });
    } else if (field == kGraphOutput && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&io](WireReader & r) { return readValueInfo(r, io.outputs); });
    } else if (field == kGraphValueInfo && wire_type == kLengthDelimited) {
      read =
        readMessage(reader, [&value_info](WireReader & r) { return readValueInfo(r, value_info); });
    } else if (field == kGraphInitializer && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&initializers, &shapes](WireReader & r) {
        std::string name;
        std::vector<int64_t> dims;
        if (!readTensor(r, name, dims)) {
          return false;
        }
        initializers.insert(name);
        shapes[name] = std::move(dims);
        return true;
      });
    } else if (field == kGraphNode && wire_type == kLengthDelimited) {
      read = readMessage(reader, [&nodes](WireReader & r) { return readNode(r, nodes); });
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

  for (const auto * tensors : {&value_info, &io.inputs}) {
    for (const auto & tensor : *tensors) {
      if (tensor.dims) {
        shapes[tensor.name] = *tensor.dims;
      }
    }
  }
  RankInference ranks(nodes, std::move(shapes));
  for (auto & output : io.outputs) {
    if (!output.dims) {
      output.rank = ranks.rank(output.name);
    }
  }
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
