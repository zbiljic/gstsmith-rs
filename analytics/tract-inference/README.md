# GStreamer Tract inference

`tractinference` is an always-in-place `video/x-raw` transform that runs one
static, single-image ONNX input through [Tract](https://github.com/sonos/tract)
and adds every raw output as a `GstTensorMeta` tensor. The video buffer,
timestamps, and unrelated metadata pass through unchanged.

The element is intentionally model-agnostic. It does not decode object
detections, apply labels or confidence thresholds, perform NMS, or draw
overlays. Connect a model-specific tensor decoder after it.

Backend-neutral model-info parsing, preprocessing, tensor caps, and
`GstTensorMeta` attachment live in the sibling `inference-common` Rust library.
Keeping the Tract backend in its own loadable plugin lets deployments install
or upgrade inference runtimes independently.

`model-channel-order=rgb` is the READY-mutable default. Set
`model-channel-order=bgr` when the model expects BGR channel order. This
changes tensor packing only: source RGB/BGR/RGBA/BGRA caps and video bytes stay
truthful and pass through unchanged. Source pixel format and model channel
order are independent.

```sh
gst-launch-1.0 \
  ... \
  ! "video/x-raw,format=RGB,width=320,height=320" \
  ! tractinference \
      model-file=model.onnx \
      model-channel-order=bgr \
  ! ...
```

## Execution provider

`execution-provider=cpu` is the default and uses Tract's CPU graph. Metal is an
opt-in macOS-only build feature:

```sh
cargo build -p gst-plugin-tract-inference --features metal

GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 \
  videotestsrc num-buffers=1 \
  ! videoconvert \
  ! "video/x-raw,format=RGB,width=320,height=320" \
  ! tractinference \
      model-file=model.onnx \
      execution-provider=metal \
  ! fakesink
```

Selecting `metal` on another platform, or from a build compiled without the
`metal` feature, fails explicitly when the element starts. It never silently
falls back to an all-CPU engine. Tract dispatches operations supported by its
Metal transform through Metal; unsupported operations may remain as CPU nodes
in the same graph. There is no automatic provider selection, device property,
or performance guarantee.

```sh
gst-launch-1.0 \
  filesrc location=input.png \
  ! pngdec \
  ! videoconvert \
  ! "video/x-raw,format=RGB,width=320,height=320" \
  ! tractinference model-file=model.onnx \
  ! my-model-tensor-decoder \
  ! fakesink
```

## Model-info contract

By default, the element reads `<model-file>.modelinfo`; `model-info-file` can
override that path. The model-info file uses format version `1.0`, preserves
the tensor section order, and must describe exactly one static batch-one image
input, any number of constant inputs, and one or more static outputs. Each
section is named after the model tensor it describes. The accepted input
video formats are RGB, BGR, RGBA, and BGRA. Image inputs may be `float32` or
`uint8`; outputs may be `float16`, `float32`, `float64`, `int8`, `int16`,
`int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `bool`. The
unsupported GStreamer 1.28 scalar encodings
(`int4`, `uint4`, and `bfloat16`) are rejected explicitly because
this backend does not expose them without conversion ambiguity.

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
have more outputs than model-info declares; undeclared outputs are neither
computed nor attached.

A model input other than the image is declared with `constant=<value>`; the
element fills the whole tensor with that value on every run. Constant inputs
take no `ranges` and may be `bool` (`true`, `false`, `1`, `0`), `float32`,
`float64`, `int32`, `int64`, or `uint8`. For example, a video encoder taking
one frame and an all-true frame mask:

```ini
[pixel_values]
id=example-pixels
type=float32
dims=1,1,3,384,384
dir=input
ranges=-1.0,1.0

[frame_mask]
id=example-frame-mask
type=bool
dims=1,1
dir=input
constant=true

[embedding]
id=example-embedding
type=float32
dims=1,1,768
dir=output
```

Every model input must be declared, and model-info must not declare tensors
the model lacks. Non-unit batch sizes, more than one image input, non-image
models, and runtime/model-info shape or scalar-type mismatches are rejected.

## Tensor-input inference

`tracttensorinference` runs a model whose inputs are tensors produced
upstream rather than video frames. It is an in-place transform accepting any
caps: the payload, timestamps, flags, and existing metadata pass through, and
the source caps gain the model's output group in the `tensors` field.

Every non-constant model-info input is looked up by its `id` across the
buffer's `GstTensorMeta`:

- all inputs present: the model runs and its declared outputs are attached as
  one new `GstTensorMeta` (input tensors stay on the buffer);
- no input present: the buffer passes through untouched, so a producer that
  only sometimes attaches inputs can share the stream;
- only some present, or a type, dimension, order, or size mismatch: element
  error.

The model-info file is the same format without the image rules: any number
of inputs of any supported type, no `ranges`, constant inputs as for images,
and dimensions that bind the model's symbolic dimensions.

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
type=bool
dims=1,400
dir=input

[scores]
id=example-scores
type=float32
dims=1,400,2
dir=output
```

`GStreamer` 1.28 cannot construct `bool` tensors (`gst_tensor_set_simple`
accepts types up to `bfloat16`), so a `type=bool` tensor travels in
`GstTensorMeta` and in tensor caps as `uint8` holding 0 or 1, in both
directions. `model-file`, `model-info-file`, and `execution-provider` behave
as for `tractinference`.

