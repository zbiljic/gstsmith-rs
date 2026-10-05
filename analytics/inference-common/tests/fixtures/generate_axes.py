"""Regenerate only the axis-contract fixtures with onnx==1.17.0."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parent


def save(name, inputs, outputs, nodes, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = helper.make_model(
        graph,
        producer_name="gstsmith axis-contract fixtures",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save(model, ROOT / f"{name}.onnx")


def value(name, dims):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


shapes = {"vector": [6], "matrix": [2, 3], "cube": [2, 1, 128]}
save(
    "tensor-axes",
    [value(name, dims) for name, dims in shapes.items()],
    [value(f"{name}_out", dims) for name, dims in shapes.items()],
    [helper.make_node("Identity", [name], [f"{name}_out"]) for name in shapes],
)

shapes = {"vector": [6], "matrix": [2, 3], "grid": [1, 2, 3]}
save(
    "image-reshape",
    [value("image", [1, 1, 2, 3])],
    [value(name, dims) for name, dims in shapes.items()],
    [helper.make_node("Reshape", ["image", f"{name}_shape"], [name]) for name in shapes],
    [
        helper.make_tensor(f"{name}_shape", TensorProto.INT64, [len(dims)], dims)
        for name, dims in shapes.items()
    ],
)
