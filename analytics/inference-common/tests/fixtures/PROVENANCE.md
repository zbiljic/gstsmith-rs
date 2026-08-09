# Test fixture provenance

`identity.onnx` is a deterministic, two-output ONNX Identity graph generated
with ONNX protobuf messages solely for this test fixture. It has
one `float32` input with dimensions `1,1,2,3` and two identical `float32`
outputs. It contains no production model data.

The adjacent model-info file is the interoperability contract used by the
test. SHA-256 checksums:

```text
2783fa57699c8499155361b3baac3c00b44d26611180ec15f10e3dc96ee886e3  identity.onnx
892f6eb51a4fddf1e2249a42bc3b2e8d198d81d923fabaa7eb77ccd0179bae51  identity.onnx.modelinfo
```
