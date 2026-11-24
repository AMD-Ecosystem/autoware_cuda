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

// A plugin library as migraphx_common_test loads it through `plugin_paths`: loading it registers
// the MIGraphX operator AutowareTestFillSeven(input, output), which fills `output` with 7.

#include <migraphx/migraphx.hpp>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
// A MIGraphX custom operator that runs on the GPU gets no output allocation from MIGraphX: the
// buffer it writes is its last input, which it returns and declares as its output alias.
struct FillSeven final : migraphx::experimental_custom_op_base
{
  std::string name() const override { return "AutowareTestFillSeven"; }

  bool runs_on_offload_target() const override { return true; }

  migraphx::shape compute_shape(migraphx::shapes inputs) const override
  {
    if (inputs.size() != 2) {
      throw std::runtime_error("AutowareTestFillSeven takes an input and an output buffer");
    }
    return inputs.back();
  }

  std::vector<std::size_t> output_alias(migraphx::shapes inputs) const override
  {
    return {inputs.size() - 1};
  }

  migraphx::argument compute(
    migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
  {
    auto output = inputs[inputs.size() - 1];
    constexpr float kSeven = 7.0F;
    uint32_t bits = 0;
    std::memcpy(&bits, &kSeven, sizeof(bits));
    if (
      hipMemsetD32Async(
        reinterpret_cast<hipDeviceptr_t>(output.data()), static_cast<int>(bits),  // NOLINT
        output.get_shape().elements(), ctx.get_queue<hipStream_t>()) != hipSuccess) {
      throw std::runtime_error("hipMemsetD32Async failed");
    }
    return output;
  }
};

FillSeven fill_seven;

const bool registered = [] {
  migraphx::register_experimental_custom_op(fill_seven);
  return true;
}();
}  // namespace

extern "C" int autoware_tensorrt_common_test_plugin_registered()
{
  return registered ? 1 : 0;
}
