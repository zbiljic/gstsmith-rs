# Inference common

`gst-inference-common` is an internal Rust library shared by gstsmith's
model-agnostic inference plugins. It reads GStreamer model-info 1.0 with
GLib's key-file parser and owns image preprocessing, engine-neutral tensor
values, tensor caps construction, and `GstTensorMeta` attachment.

Its deterministic ONNX/model-info fixtures are also the shared compatibility
contract used by backend parity tests: `identity` covers the basic image
contract, `masked-frames` covers model initializers, symbolic dimensions bound
by model-info, and output subsets, and `masked-sequence` covers upstream
tensor inputs shared by ORT and Tract, including an upstream mask and scale.

Preprocessing decodes truthful RGB, BGR, RGBA, or BGRA source pixels into
semantic red, green, and blue values, then packs them in the channel order
requested by the backend. The default model order is RGB; BGR is an explicit
opt-in. When model-info declares three normalization ranges, their order is
always semantic R, G, B regardless of the model's channel order.
Backend elements expose this choice as `model-channel-order`; set
`model-channel-order=bgr` on either element for BGR channel order.

It is an `rlib`, not a loadable GStreamer plugin. Backend crates link it
statically and remain independently installable. Public GStreamer factories
and backend-specific runtime configuration belong in those plugin crates.

The following contract applies to both [ORT inference](../ort-inference/README.md)
and [Tract inference](../tract-inference/README.md). Their READMEs cover build
instructions, pipeline examples, and execution-provider configuration.

## Model-info contract

By default, both inference elements read `<model-file>.modelinfo`;
`model-info-file` can override that path. The file follows the upstream
[GStreamer model-info format](https://gstreamer.freedesktop.org/documentation/analytics/GstAnalyticsModelInfo.html),
version `1.0`. Format `1.1` is not supported yet. Tensor section order is
preserved. In video mode, each element supports exactly one static batch-one
image input and one or more static outputs. Each section is named after
the model tensor it describes. The accepted input
video formats are RGB, BGR, RGBA, and BGRA. Image inputs may be `float32` or
`uint8`; outputs may be `float16`, `float32`, `float64`, `int8`, `int16`,
`int32`, `int64`, `uint8`, `uint16`, `uint32`, or `uint64`. Unsupported
encodings, including `bool`, `int4`, `uint4`, and `bfloat16`, are rejected
explicitly. Neither element substitutes another tensor type.

```ini
[modelinfo]
version=1.0
group-id=example-model-output

[image]
id=example-input
type=float32
dims=1,320,320,3
dir=input
ranges=0.0,1.0

[scores]
id=example-scores
type=float32
dims=1,1000
dir=output
```

Input dimensions choose HWC (`1,H,W,3`) or CHW (`1,3,H,W`) packing. Further
unit dimensions may follow the batch dimension, such as a frame count
(`1,1,3,H,W`). `ranges`
maps byte pixels into the model’s range per channel (one range applies to all
channels; three ranges are always semantic R, G, B, including when
`model-channel-order=bgr`). Source caps retain the input
video structure and add a `tensors` group keyed by `group-id`; each
`tensor/strided` descriptor contains the declared dimensions, order, type, and
tensor ID. Each declared output becomes a separate buffer in `GstTensorMeta`.

Model-info dimensions are authoritative: they bind the model's symbolic
(dynamic) dimensions, while fixed model dimensions must match. A model may
have more outputs than model-info declares; only declared outputs are
requested and attached.

Fixed constants belong in the model graph as initializers. Additional runtime
inputs must be supplied upstream in tensor-input mode. Unknown model-info
fields are ignored, as upstream specifies.

Every model input must be declared, and model-info must not declare tensors
the model lacks. Non-unit batch sizes and runtime/model-info shape or
scalar-type mismatches are rejected. Video mode additionally rejects more
than one image input and non-image models.

## Tensor-input inference

Set `input-mode=tensor-meta` on either inference element to run a model on
tensors prepared upstream. Any producer can attach the inputs: preprocessing,
feature extraction, a custom source, or another
inference element. This mode accepts any carrier caps: the payload,
timestamps, flags, and existing metadata pass through, and the source caps
gain the model's output group in the `tensors` field. Tensor bytes come from
each `GstTensor`'s data buffer, not the carrier's payload.

`input-mode` is mutable in NULL or READY and defaults to `video`. Video mode
accepts RGB/BGR/RGBA/BGRA frames at the model's image dimensions and continues
to preprocess pixels even if tensors are already attached. `model-channel-order`
only affects video mode. Execution-provider selection applies to both modes.

Every model-info input is looked up by its `id` across the
buffer's `GstTensorMeta` instances, regardless of tensor order:

- all inputs present: the model runs and its declared outputs are attached as
  one new `GstTensorMeta` (input tensors stay on the buffer);
- no input present: the buffer passes through untouched, so a producer that
  only sometimes attaches inputs can share the stream;
- only some present, a duplicate required ID, or a type, dimension, order, or
  size mismatch: element error.

Unrelated tensors are ignored and retained. The model-info section name is
the model's input name; its `id` selects the upstream tensor. Multiple selected
tensors feed one model invocation per buffer, not one invocation per tensor.
For example, a secondary model can map `[features]` to `id=primary-features`
and `[mask]` to `id=prepared-mask`, leaving primary detection tensors untouched.
Use distinct output IDs and `group-id` values for each stage. `group-id`
describes outputs in caps; it does not disambiguate duplicate input IDs.

This supports primary/secondary model chaining when the tensor contracts
match. Per-object classification after detection additionally needs crop
preprocessing, repeated execution or batching, and association of each output
with its object. That workflow is not implemented by input selection alone.

The model-info file is the same format without the image rules: one or more
inputs of any supported type and dimensions that bind the model's symbolic
dimensions. Inputs are already preprocessed, so `ranges` is unused in this
mode. All tensors must use row-major dimension order and retain a batch
dimension of one. Tensor byte sizes must fit the platform allocation limit.

```ini
[modelinfo]
version=1.0
group-id=example-sequence-model

[input_embs]
id=example-embeddings
type=float32
dims=1,400,768
dir=input

[attention_mask]
id=example-mask
type=uint8
dims=1,400
dir=input

[scores]
id=example-scores
type=float32
dims=1,400,2
dir=output
```

In this example the model's mask input must actually be `uint8` (the model
can cast it internally). `model-file`, `model-info-file`, and
`execution-provider` behave as in video mode.
