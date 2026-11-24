#!/usr/bin/env python3
"""Regenerate the models and references used by migraphx_common_test.

conv_network.onnx is a small classifier with a dynamic batch: a 3x3 convolution, ReLU, global
average pooling and a fully connected layer. It returns two outputs, `logits` and then `features`,
so the order of the outputs is observable. Variants of it check that a cached program is rebuilt
when the model changes (other weights, a renamed input, another input size) and that output
dimensions the model leaves symbolic are reported once known. activations.onnx checks that
operations can be kept out of fp16, rows_relu.onnx and rows.onnx take inputs with no rows, and
custom_op_network.onnx calls an operator that a plugin library registers. Each reference file holds the float32 outputs for the input the test builds
(see `inputValue` in the test), computed here with numpy.

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

here = pathlib.Path(__file__)


def test_input(shape):
    """The test input: element i of the flattened tensor is ((i * 37) % 101) / 101 - 0.5."""
    count = int(np.prod(shape))
    return (((np.arange(count) * 37) % 101) / 101.0 - 0.5).astype(np.float32).reshape(shape)


def save(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, str(here.with_name(name)))


def save_reference(outputs, name):
    reference = np.concatenate([o.ravel() for o in outputs]).astype("<f4")
    reference.tofile(str(here.with_name(name)))


def conv_weights(seed):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-0.5, 0.5, (FILTERS, CHANNELS, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, (FILTERS,)).astype(np.float32)
    fc_weight = rng.uniform(-1.0, 1.0, (FILTERS, CLASSES)).astype(np.float32)
    fc_bias = rng.uniform(-0.1, 0.1, (CLASSES,)).astype(np.float32)
    return weight, bias, fc_weight, fc_bias


def conv_network(
    name,
    seed=0,
    input_name="image",
    size=SIZE,
    batch=BATCH,
    logits_dims=(BATCH, CLASSES),
    features_dims=(BATCH, FILTERS),
):
    weight, bias, fc_weight, fc_bias = conv_weights(seed)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Conv", [input_name, "weight", "bias"], ["conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node("GlobalAveragePool", ["relu"], ["pool"]),
            helper.make_node("Flatten", ["pool"], ["features"]),
            helper.make_node("Gemm", ["features", "fc_weight", "fc_bias"], ["logits"]),
        ],
        name="conv_network",
        inputs=[
            helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [batch, CHANNELS, size, size]
            )
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(logits_dims)),
            helper.make_tensor_value_info("features", TensorProto.FLOAT, list(features_dims)),
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
    save(model, name)


def conv_reference(name, seed=0, size=SIZE):
    weight, bias, fc_weight, fc_bias = conv_weights(seed)
    image = test_input((REFERENCE_BATCH, CHANNELS, size, size))
    padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros((REFERENCE_BATCH, FILTERS, size, size), np.float64)
    for y in range(3):
        for x in range(3):
            window = padded[:, :, y : y + size, x : x + size]
            conv += np.einsum("nchw,fc->nfhw", window, weight[:, :, y, x])
    conv += bias[None, :, None, None]
    features = np.maximum(conv, 0).mean(axis=(2, 3))
    logits = features @ fc_weight + fc_bias
    save_reference([logits, features], name)


conv_network("conv_network.onnx")
conv_reference("conv_network_reference.bin")

# The same inputs and outputs with other weights.
conv_network("conv_network_retrained.onnx", seed=1)
conv_reference("conv_network_retrained_reference.bin", seed=1)
# The input renamed; outputs as conv_network.onnx.
conv_network("conv_network_renamed.onnx", input_name="pixels")
# A 16x16 input.
conv_network("conv_network_resized.onnx", size=16)
conv_reference("conv_network_resized_reference.bin", size=16)
# Output dimensions left symbolic, with a static input and with a dynamic batch.
conv_network(
    "conv_network_static_symbolic.onnx",
    batch=2,
    logits_dims=("unk__0", "unk__1"),
    features_dims=("unk__2", FILTERS),
)
conv_network(
    "conv_network_dynamic_symbolic.onnx",
    logits_dims=(BATCH, "classes"),
    features_dims=("unk__0", "unk__1"),
)

# Sigmoid and softmax of a matrix product, which MIGraphX computes in fp16 unless told otherwise.
ACTIVATION_ROWS = 4
ACTIVATION_COLUMNS = 64
activation_weight = (
    np.random.default_rng(2)
    .uniform(-0.5, 0.5, (ACTIVATION_COLUMNS, ACTIVATION_COLUMNS))
    .astype(np.float32)
)
activations = helper.make_graph(
    nodes=[
        helper.make_node("MatMul", ["x", "weight"], ["product"]),
        helper.make_node("Sigmoid", ["product"], ["sigmoid"]),
        helper.make_node("Softmax", ["product"], ["softmax"], axis=1),
    ],
    name="activations",
    inputs=[
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [ACTIVATION_ROWS, ACTIVATION_COLUMNS])
    ],
    outputs=[
        helper.make_tensor_value_info(
            "sigmoid", TensorProto.FLOAT, [ACTIVATION_ROWS, ACTIVATION_COLUMNS]
        ),
        helper.make_tensor_value_info(
            "softmax", TensorProto.FLOAT, [ACTIVATION_ROWS, ACTIVATION_COLUMNS]
        ),
    ],
    initializer=[numpy_helper.from_array(activation_weight, "weight")],
)
model = helper.make_model(activations, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
save(model, "activations.onnx")
product = test_input((ACTIVATION_ROWS, ACTIVATION_COLUMNS)).astype(np.float64) @ activation_weight
sigmoid = 1.0 / (1.0 + np.exp(-product))
softmax = np.exp(product - product.max(axis=1, keepdims=True))
softmax /= softmax.sum(axis=1, keepdims=True)
save_reference([sigmoid, softmax], "activations_reference.bin")

# Per-row ReLU of x [rows, 4]; rows.onnx also returns the column sums, which are not empty when
# there are no rows.
ROWS = "rows"
row_inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, 4])]
relu_output = helper.make_tensor_value_info("relu", TensorProto.FLOAT, [ROWS, 4])
for name, nodes, outputs, initializers in (
    ("rows_relu.onnx", [helper.make_node("Relu", ["x"], ["relu"])], [relu_output], []),
    (
        "rows.onnx",
        [
            helper.make_node("Relu", ["x"], ["relu"]),
            helper.make_node("ReduceSum", ["x", "axes"], ["column_sum"], keepdims=0),
        ],
        [relu_output, helper.make_tensor_value_info("column_sum", TensorProto.FLOAT, [4])],
        [numpy_helper.from_array(np.array([0], np.int64), "axes")],
    ),
):
    graph = helper.make_graph(nodes, "rows", row_inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    save(model, name)

# y = AutowareTestFillSeven(x) + x = x + 7, where the test plugin library registers the operator.
custom = helper.make_graph(
    nodes=[
        helper.make_node("AutowareTestFillSeven", ["x"], ["seven_more"]),
        helper.make_node("Add", ["seven_more", "x"], ["y"]),
    ],
    name="custom_op_network",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [16])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [16])],
)
model = helper.make_model(custom, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.save(model, str(here.with_name("custom_op_network.onnx")))

print("wrote the test models and references")
