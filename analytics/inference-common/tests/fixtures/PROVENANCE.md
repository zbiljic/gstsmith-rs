# Test fixture provenance

`identity.onnx` is a deterministic, two-output ONNX Identity graph
generated with ONNX protobuf messages solely for this test fixture. It
has one `float32` input with dimensions `1,1,2,3` and two identical
`float32` outputs. It contains no production model data.

`masked-frames.onnx` is a deterministic ONNX graph generated with the
`onnx.helper` API solely for this test fixture. It has a `float32` input
`x` with dimensions `batch,frames,1,2,3` (two symbolic dimensions), a
`bool` initializer `mask` with dimensions `1,1` and value `true`, an
output `y = x * mask` (the mask broadcast over the image), and a second
output `z = x`. Its model-info binds the symbolic dimensions to one and
declares only `y`, so `z` must not be attached. It contains no
production model data.

`masked-sequence.onnx` is a deterministic ONNX graph generated with the
`onnx.helper` API solely for the tensor-input fixture. It has a
`float32` input `x` with dimensions `batch,seq,2`, a `uint8` input
`mask` with dimensions `batch,seq`, a `float32` input `scale` with
dimension `1`, an output `y = x * mask * scale` (the mask broadcast over
the last axis), and a second output `z = x`. Its model-info binds the
symbolic dimensions to `1,3`, takes `x`, `mask`, and `scale` from
upstream tensors, and declares only `y`. It contains no production model
data.

The adjacent model-info files are the interoperability contracts used by
the tests. SHA-256 checksums:

```text
2783fa57699c8499155361b3baac3c00b44d26611180ec15f10e3dc96ee886e3  identity.onnx
892f6eb51a4fddf1e2249a42bc3b2e8d198d81d923fabaa7eb77ccd0179bae51  identity.onnx.modelinfo
8be18cbb997c51eef88a4e7386c2f6d9953a2ffeb7279510217c99c5d2c0968d  masked-frames.onnx
f851e30804cb3a9d8b66c197a7d75329e49d725b7efa4e5d39fecb5a481f4748  masked-frames.onnx.modelinfo
774162c53152c8aa958f1b4453e42e2579ef2558d51b91593384a5aeb3783a69  masked-sequence.onnx
89abf2849fd5a6f650580294d312494ddc2d2ec1fea3e79e56b17be2ed998487  masked-sequence.onnx.modelinfo
```
