# Test fixture provenance

`identity.onnx` is a deterministic, two-output ONNX Identity graph generated
with ONNX protobuf messages solely for this test fixture. It has
one `float32` input with dimensions `1,1,2,3` and two identical `float32`
outputs. It contains no production model data.

`masked-frames.onnx` is a deterministic ONNX graph generated with the
`onnx.helper` API solely for this test fixture. It has a `float32` input `x`
with dimensions `batch,frames,1,2,3` (two symbolic dimensions), a `bool`
input `mask` with dimensions `batch,frames`, an output `y = x * mask` (the
mask broadcast over the image), and a second output `z = x`. Its model-info
binds the symbolic dimensions to one, declares `mask` as a constant input,
and declares only `y`, so `z` must not be computed or attached. It contains
no production model data.

The adjacent model-info files are the interoperability contracts used by the
tests. SHA-256 checksums:

```text
2783fa57699c8499155361b3baac3c00b44d26611180ec15f10e3dc96ee886e3  identity.onnx
892f6eb51a4fddf1e2249a42bc3b2e8d198d81d923fabaa7eb77ccd0179bae51  identity.onnx.modelinfo
50d39ad872435a10a5e78a07ae894606c37debcd722a979f6a5768cda375b417  masked-frames.onnx
5f791856f1080f97aad1f6ef34c74c26e073a142d198d6e4aa8d75447ce5af9d  masked-frames.onnx.modelinfo
```
