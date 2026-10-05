# GStreamer Tract inference

`tractinference` is an always-in-place transform that runs ONNX models
through [Tract](https://github.com/sonos/tract) and attaches the outputs declared
in model-info as `GstTensorMeta`. The default `input-mode=video` preprocesses
one static image input from `video/x-raw`. Set `input-mode=tensor-meta` to use
[tensors prepared upstream](../inference-common/README.md#tensor-input-inference).
The carrier buffer, timestamps, and existing metadata pass through unchanged.

The element is intentionally model-agnostic. It does not decode object
detections, apply labels or confidence thresholds, perform NMS, or draw
overlays. Connect a model-specific tensor decoder after it.

Backend-neutral model-info parsing, preprocessing, tensor caps, and
`GstTensorMeta` attachment live in the sibling
[`inference-common` Rust library](../inference-common/README.md).
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

export GST_PLUGIN_PATH="$PWD/target/debug"
gst-launch-1.0 \
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

## Shared inference contract

See the shared [model-info contract](../inference-common/README.md#model-info-contract)
for sidecar files, supported tensor types, dimensions, and preprocessing, and
the [tensor-input contract](../inference-common/README.md#tensor-input-inference)
for upstream tensor selection, caps, errors, and model chaining.

`input-mode` is mutable in NULL or READY. Provider selection applies to both
input modes; `model-channel-order` applies only to video mode.

Tract requires concrete model-info dimensions; wildcard declarations are
rejected at startup. Tensor inputs and outputs may have non-unit leading
dimensions. Batch-one image rules apply only to video input. See the shared
[compatibility notes](../inference-common/README.md#compatibility)
for the scope of the fixture tests. ORT's inspection example does not validate
Tract graph support.
