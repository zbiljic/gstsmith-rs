# Convolution fixture

`metal-conv.onnx` is a generated ONNX opset 13 graph with one 1x1 convolution.
Its float32 NCHW input is `[1,3,2,2]`, weights are `[1,2,3]`, bias is `0.5`,
and output is `[1,1,2,2]`. It is shared by the Tract Metal and ORT CoreML tests.

The convolution explicitly sets `pads=[0,0,0,0]`. This preserves the ONNX
default while avoiding an ORT MLProgram conversion that omits the CoreML
`pad` input, which CoreML rejects on the tested runtime.

The padding attribute was added by decoding and re-encoding the original
fixture with `protoc` and the Apache-2.0 ONNX schema shipped in Tract 0.23.8
(`protos/onnx/onnx.proto3`). The added node attribute is:

```text
attribute { name: "pads" ints: 0 ints: 0 ints: 0 ints: 0 type: INTS }
```

SHA-256:

```text
37b7e1d75b486c5dbdc18b76fa39d9a02128ed79906f9a73807d0deb0fba233a  metal-conv.onnx
```
