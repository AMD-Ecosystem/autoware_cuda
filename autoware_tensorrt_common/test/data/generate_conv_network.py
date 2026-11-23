#!/usr/bin/env python3
"""Regenerate conv_network.onnx and conv_network_reference.bin, used by migraphx_common_test.

The graph is a small classifier with a dynamic batch: a 3x3 convolution, ReLU, global average
pooling and a fully connected layer. It returns two outputs, `logits` and then `features`, so the
order of the outputs is observable. The reference file holds the float32 outputs for the
input batch the test builds (see `inputValue` in the test), computed here with numpy.

Writes the files next to this script, so it can be run from anywhere.

Usage: python3 generate_conv_network.py
"""

# cspell:ignore opset, opsetid

import pathlib

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

BATCH = "batch"
CHANNELS = 3
SIZE = 8
FILTERS = 4
CLASSES = 2
REFERENCE_BATCH = 16

rng = np.random.default_rng(0)
weight = rng.uniform(-0.5, 0.5, (FILTERS, CHANNELS, 3, 3)).astype(np.float32)
bias = rng.uniform(-0.1, 0.1, (FILTERS,)).astype(np.float32)
fc_weight = rng.uniform(-1.0, 1.0, (FILTERS, CLASSES)).astype(np.float32)
fc_bias = rng.uniform(-0.1, 0.1, (CLASSES,)).astype(np.float32)

graph = helper.make_graph(
    nodes=[
        helper.make_node("Conv", ["image", "weight", "bias"], ["conv"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["features"]),
        helper.make_node("Gemm", ["features", "fc_weight", "fc_bias"], ["logits"]),
    ],
    name="conv_network",
    inputs=[
        helper.make_tensor_value_info("image", TensorProto.FLOAT, [BATCH, CHANNELS, SIZE, SIZE])
    ],
    outputs=[
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, CLASSES]),
        helper.make_tensor_value_info("features", TensorProto.FLOAT, [BATCH, FILTERS]),
    ],
    initializer=[
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(fc_weight, "fc_weight"),
        numpy_helper.from_array(fc_bias, "fc_bias"),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
here = pathlib.Path(__file__)
onnx.save(model, str(here.with_name("conv_network.onnx")))

# The test input: element i of the flattened batch is ((i * 37) % 101) / 101 - 0.5.
count = REFERENCE_BATCH * CHANNELS * SIZE * SIZE
image = (((np.arange(count) * 37) % 101) / 101.0 - 0.5).astype(np.float32)
image = image.reshape(REFERENCE_BATCH, CHANNELS, SIZE, SIZE)

padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
conv = np.zeros((REFERENCE_BATCH, FILTERS, SIZE, SIZE), np.float64)
for y in range(3):
    for x in range(3):
        window = padded[:, :, y : y + SIZE, x : x + SIZE]
        conv += np.einsum("nchw,fc->nfhw", window, weight[:, :, y, x])
conv += bias[None, :, None, None]
features = np.maximum(conv, 0).mean(axis=(2, 3))
logits = features @ fc_weight + fc_bias

reference = np.concatenate([logits.ravel(), features.ravel()]).astype("<f4")
reference.tofile(str(here.with_name("conv_network_reference.bin")))
print("wrote conv_network.onnx and conv_network_reference.bin")
