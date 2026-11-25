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

#ifndef ONNX_MODEL_IO_HPP_
#define ONNX_MODEL_IO_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace autoware
{
namespace tensorrt_common
{
/**
 * @brief A graph input or output as the ONNX model declares it.
 */
struct OnnxTensor
{
  std::string name;
  //!< ONNX TensorProto.DataType value, 0 when the model does not say.
  int32_t elem_type{0};
  //!< Dimensions, -1 for a symbolic or unknown one; empty for a scalar, nullopt when the model
  //!< declares no shape.
  std::optional<std::vector<int64_t>> dims;
  //!< For a graph output the model gives no shape: its rank as inferred from the graph's nodes,
  //!< or nullopt when it cannot be.
  std::optional<std::size_t> rank;
};

/**
 * @brief The inputs and outputs of an ONNX model's main graph, in the order the model lists them.
 *
 * Graph inputs that are initializers (weights) are not inputs here.
 */
struct OnnxModelIO
{
  std::vector<OnnxTensor> inputs;
  std::vector<OnnxTensor> outputs;
};

/**
 * @brief Read the graph inputs and outputs of an ONNX model file.
 *
 * Besides the graph's inputs and outputs, only what infers the rank of an output with no shape is
 * decoded: the nodes, and the names and dimensions of initializers and value_info. Weights are
 * skipped without being read into memory.
 *
 * @param[in] path ONNX model path.
 * @return The model IO, or nullopt if the file cannot be read or is not an ONNX model.
 */
std::optional<OnnxModelIO> readOnnxModelIO(const std::string & path);
}  // namespace tensorrt_common
}  // namespace autoware

#endif  // ONNX_MODEL_IO_HPP_
