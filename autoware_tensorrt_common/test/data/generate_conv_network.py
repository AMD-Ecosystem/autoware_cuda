#!/usr/bin/env python3
"""Regenerate the models and references used by migraphx_common_test.

conv_network.onnx is a small classifier with a dynamic batch: a 3x3 convolution, ReLU, global
average pooling and a fully connected layer. It returns two outputs, `logits` and then `features`,
so the order of the outputs is observable. Variants of it check that a cached program is rebuilt
when the model changes (other weights, a renamed input, another input size) and that output
dimensions the model leaves symbolic are reported once known; the program of the variant with
three classes stands in for a foreign program cached under the key of conv_network.onnx.
pool_network.onnx has output dimensions that follow a dynamic one through a floor division, and
shapeless_output.onnx, shapeless_output_flat.onnx, squeezed.onnx, row_sums.onnx and the
untold_rank models an output the model gives no shape, as does output_ranks.onnx for each way an
output rank is inferred from the model. The scalar_sum models return a scalar,
with and without a shape in the model, and scalar_scale.onnx takes one. activations.onnx
and scaled_sigmoid.onnx check that operations can be kept out of fp16, rows_scaled.onnx and
rows.onnx take inputs with no rows, and custom_op_network.onnx calls an operator that a plugin
library registers. Each reference file holds the float32 outputs for the input the test builds (see
`inputValue` in the test), computed here with numpy.

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


def conv_weights(seed, classes=CLASSES):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-0.5, 0.5, (FILTERS, CHANNELS, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, (FILTERS,)).astype(np.float32)
    fc_weight = rng.uniform(-1.0, 1.0, (FILTERS, classes)).astype(np.float32)
    fc_bias = rng.uniform(-0.1, 0.1, (classes,)).astype(np.float32)
    return weight, bias, fc_weight, fc_bias


def conv_network(
    name,
    seed=0,
    input_name="image",
    size=SIZE,
    batch=BATCH,
    logits_dims=(BATCH, CLASSES),
    features_dims=(BATCH, FILTERS),
    classes=CLASSES,
):
    weight, bias, fc_weight, fc_bias = conv_weights(seed, classes)
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
# Three classes.
conv_network("conv_network_three_classes.onnx", logits_dims=(BATCH, 3), classes=3)
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

# A division by a constant before a sigmoid. MIGraphX turns the division into a multiplication
# before it converts operations to fp16.
SCALED_COLUMNS = 64
scaled = helper.make_graph(
    nodes=[
        helper.make_node("Div", ["x", "divisor"], ["quotient"]),
        helper.make_node("Sigmoid", ["quotient"], ["y"]),
    ],
    name="scaled_sigmoid",
    inputs=[
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [ACTIVATION_ROWS, SCALED_COLUMNS])
    ],
    outputs=[
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [ACTIVATION_ROWS, SCALED_COLUMNS])
    ],
    initializer=[numpy_helper.from_array(np.full((SCALED_COLUMNS,), 0.3, np.float32), "divisor")],
)
model = helper.make_model(scaled, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
save(model, "scaled_sigmoid.onnx")
quotient = test_input((ACTIVATION_ROWS, SCALED_COLUMNS)).astype(np.float64) / np.float32(0.3)
save_reference([1.0 / (1.0 + np.exp(-quotient))], "scaled_sigmoid_reference.bin")

# Max pooling of x [1, 2, height, 8] over 2x2 windows with stride 2, and over 1x1 windows with a
# stride of 300 rows: the output heights are floor divisions of the dynamic height.
pool = helper.make_graph(
    nodes=[
        helper.make_node("MaxPool", ["x"], ["halved"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("MaxPool", ["x"], ["sparse"], kernel_shape=[1, 1], strides=[300, 1]),
    ],
    name="pool_network",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "height", 8])],
    outputs=[
        helper.make_tensor_value_info("halved", TensorProto.FLOAT, [1, 2, "halved_height", 4]),
        helper.make_tensor_value_info("sparse", TensorProto.FLOAT, [1, 2, "sparse_height", 8]),
    ],
)
model = helper.make_model(pool, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
save(model, "pool_network.onnx")

# Per-row ReLU of x [rows, 4]. rows_scaled.onnx also returns b * 2 for a second input b [4], and
# rows.onnx the column sums of x: outputs that are not empty when there are no rows.
ROWS = "rows"
row_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, 4])
relu_output = helper.make_tensor_value_info("relu", TensorProto.FLOAT, [ROWS, 4])
for name, nodes, inputs, outputs, initializers in (
    (
        "rows_scaled.onnx",
        [
            helper.make_node("Relu", ["x"], ["relu"]),
            helper.make_node("Mul", ["b", "two"], ["scaled"]),
        ],
        [row_input, helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])],
        [relu_output, helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [4])],
        [numpy_helper.from_array(np.array([2.0], np.float32), "two")],
    ),
    (
        "rows.onnx",
        [
            helper.make_node("Relu", ["x"], ["relu"]),
            helper.make_node("ReduceSum", ["x", "axes"], ["column_sum"], keepdims=0),
        ],
        [row_input],
        [relu_output, helper.make_tensor_value_info("column_sum", TensorProto.FLOAT, [4])],
        [numpy_helper.from_array(np.array([0], np.int64), "axes")],
    ),
):
    graph = helper.make_graph(nodes, "rows", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    save(model, name)

# Outputs the model gives no shape, of x [rows, 4]: its ReLU, the same flattened to [rows * 4],
# and Squeeze with no axes, which drops the rows dimension only when there is one row. The ONNX
# checker requires a shape for graph outputs, so these are saved unchecked.
for name, nodes, initializers in (
    ("shapeless_output.onnx", [helper.make_node("Relu", ["x"], ["y"])], []),
    (
        "shapeless_output_flat.onnx",
        [helper.make_node("Reshape", ["x", "flat"], ["y"])],
        [numpy_helper.from_array(np.array([-1], np.int64), "flat")],
    ),
    ("squeezed.onnx", [helper.make_node("Squeeze", ["x"], ["y"])], []),
):
    graph = helper.make_graph(
        nodes,
        "shapeless_output",
        [row_input],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(here.with_name(name)))

# The sum of every element of x [rows, 4] or [3, 4], a scalar, which the model declares as one
# or gives no shape.
for name, rows, y_dims in (
    ("scalar_sum.onnx", ROWS, []),
    ("scalar_sum_static.onnx", 3, []),
    ("scalar_sum_shapeless.onnx", ROWS, None),
    ("scalar_sum_shapeless_static.onnx", 3, None),
):
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["x"], ["y"], keepdims=0)],
        "scalar_sum",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, y_dims)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    if y_dims is None:
        onnx.save(model, str(here.with_name(name)))
    else:
        save(model, name)

# y = x * s for x [rows, 4] and a scalar s.
graph = helper.make_graph(
    [helper.make_node("Mul", ["x", "s"], ["y"])],
    "scalar_scale",
    [row_input, helper.make_tensor_value_info("s", TensorProto.FLOAT, [])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [ROWS, 4])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
save(model, "scalar_scale.onnx")

# Row sums of x [rows, 4], given no shape: [rows], which has one element when there is one row,
# as a scalar does. Einsum "ij->i" computes the same, an operator whose output rank the backend
# does not infer from the model, here also with x [1, 4].
graph = helper.make_graph(
    [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=0)],
    "row_sums",
    [row_input],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    initializer=[numpy_helper.from_array(np.array([1], np.int64), "axes")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.save(model, str(here.with_name("row_sums.onnx")))
for name, rows in (("untold_rank_rows.onnx", ROWS), ("untold_rank.onnx", 1)):
    graph = helper.make_graph(
        [helper.make_node("Einsum", ["x"], ["y"], equation="ij->i")],
        "untold_rank",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(here.with_name(name)))

# Outputs given no shape, one or two per rule by which the backend infers a rank from the model, of
# x [2, 3, 4]. The ranks the test expects are those ONNX shape inference gives.
RANKS = {
    "relu": 3,
    "transposed": 3,
    "split_0": 3,
    "split_1": 3,
    "top_values": 3,
    "top_indices": 3,
    "sum": 3,
    "where": 3,
    "kept_sum": 3,
    "column_means": 2,
    "total": 0,
    "arg_max": 2,
    "squeezed": 2,
    "unsqueezed": 5,
    "reshaped": 2,
    "expanded": 4,
    "filled": 2,
    "flat": 2,
    "gemm": 2,
    "vector_product": 2,
    "batched_product": 3,
    "picked": 2,
    "gathered": 4,
    "shape": 1,
    "size": 0,
    "constant": 2,
    "nonzero": 2,
    "range": 1,
}
tensor = numpy_helper.from_array
rank_graph = helper.make_graph(
    [
        helper.make_node("Relu", ["x"], ["relu"]),
        helper.make_node("Transpose", ["x"], ["transposed"], perm=[2, 0, 1]),
        helper.make_node("Split", ["x"], ["split_0", "split_1"], axis=2),
        helper.make_node("TopK", ["x", "k"], ["top_values", "top_indices"]),
        helper.make_node("Add", ["x", "row"], ["sum"]),
        helper.make_node("Where", ["mask", "x", "row"], ["where"]),
        helper.make_node("ReduceSum", ["x", "axis_1"], ["kept_sum"]),
        helper.make_node("ReduceMean", ["x"], ["column_means"], axes=[-1], keepdims=0),
        helper.make_node("ReduceSum", ["x"], ["total"], keepdims=0),
        helper.make_node("ArgMax", ["x"], ["arg_max"], keepdims=0),
        helper.make_node("Squeeze", ["kept_sum", "axis_1"], ["squeezed"]),
        helper.make_node("Unsqueeze", ["x", "axes_0_4"], ["unsqueezed"]),
        helper.make_node("Reshape", ["x", "shape_6_4"], ["reshaped"]),
        helper.make_node("Expand", ["x", "shape_1_2_3_4"], ["expanded"]),
        helper.make_node("ConstantOfShape", ["shape_6_4"], ["filled"]),
        helper.make_node("Flatten", ["x"], ["flat"]),
        helper.make_node("Gemm", ["flat", "weight_12_5"], ["gemm"]),
        helper.make_node("MatMul", ["row", "weight_4_5"], ["row_product"]),
        helper.make_node("MatMul", ["x", "weight_4_5"], ["batched_product"]),
        helper.make_node("Unsqueeze", ["row_product", "axes_0"], ["vector_product"]),
        helper.make_node("Gather", ["x", "index"], ["picked"]),
        helper.make_node("Gather", ["x", "indices_2_2"], ["gathered"]),
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("Size", ["x"], ["size"]),
        helper.make_node(
            "Constant", [], ["constant"], value=tensor(np.zeros((2, 2), np.float32), "c")
        ),
        helper.make_node("NonZero", ["x"], ["nonzero"]),
        helper.make_node("Range", ["start", "limit", "delta"], ["range"]),
    ],
    "ranks",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
    [helper.make_tensor_value_info(name, TensorProto.UNDEFINED, None) for name in RANKS],
    initializer=[
        tensor(np.array([2], np.int64), "k"),
        tensor(np.ones(4, np.float32), "row"),
        tensor(np.ones(4, bool), "mask"),
        tensor(np.array([1], np.int64), "axis_1"),
        tensor(np.array([0, 4], np.int64), "axes_0_4"),
        tensor(np.array([0], np.int64), "axes_0"),
        tensor(np.array([6, 4], np.int64), "shape_6_4"),
        tensor(np.array([1, 2, 3, 4], np.int64), "shape_1_2_3_4"),
        tensor(np.ones((12, 5), np.float32), "weight_12_5"),
        tensor(np.ones((4, 5), np.float32), "weight_4_5"),
        tensor(np.array(1, np.int64), "index"),
        tensor(np.zeros((2, 2), np.int64), "indices_2_2"),
        tensor(np.array(0.0, np.float32), "start"),
        tensor(np.array(3.0, np.float32), "limit"),
        tensor(np.array(1.0, np.float32), "delta"),
    ],
)
model = helper.make_model(rank_graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph.output
assert {o.name: len(o.type.tensor_type.shape.dim) for o in inferred} == RANKS
onnx.save(model, str(here.with_name("output_ranks.onnx")))

# y = AutowareTestFillSeven(x, buffer) + x = x + 7, where the test plugin library registers the
# operator. As a MIGraphX custom operator that runs on the GPU must, it writes its result into
# one of its inputs, here the buffer passed last. The `buffer` graph input exists only because
# nothing in an ONNX graph can supply that allocation; a model written for a TensorRT plugin has no
# such input.
custom = helper.make_graph(
    nodes=[
        helper.make_node("AutowareTestFillSeven", ["x", "buffer"], ["sevens"]),
        helper.make_node("Add", ["sevens", "x"], ["y"]),
    ],
    name="custom_op_network",
    inputs=[
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [16]),
        helper.make_tensor_value_info("buffer", TensorProto.FLOAT, [16]),
    ],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [16])],
)
model = helper.make_model(custom, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.save(model, str(here.with_name("custom_op_network.onnx")))

print("wrote the test models and references")
