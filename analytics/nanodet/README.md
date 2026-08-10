# NanoDet tensor decoder

`nanodettensordec` decodes supported NanoDet-m and NanoDet-Plus output tensors
and attaches object-detection and classification metadata to the video buffer.

## Supported tensor contracts

The decoder selects the contract from the negotiated video size and tensor
shape. Model width multipliers such as 1.0x and 1.5x change the network internals
but not these output contracts, so they require no decoder property or profile.

| Family | Input | Output | Feature strides |
| --- | ---: | ---: | --- |
| NanoDet-m | 320x320 | `[1,2100,112]` | 8, 16, 32 |
| NanoDet-Plus | 320x320 | `[1,2125,112]` | 8, 16, 32, 64 |
| NanoDet-m | 416x416 | `[1,3549,112]` | 8, 16, 32 |
| NanoDet-Plus | 416x416 | `[1,3598,112]` | 8, 16, 32, 64 |

Every row contains 80 post-sigmoid class probabilities followed by four sets of
eight distribution-regression bins (`reg-max=7`).

The decoder accepts row-major Float32 and Float16 output tensors. Quantized
models are compatible when the inference backend exposes their final output as
Float32 or Float16. Raw INT8 outputs are unsupported because the tensor metadata
does not include dequantization parameters.

The decoder uses the tensor ID `nanodet-output` by default. A buffer without the
configured tensor passes through unchanged; an incompatible tensor produces a
streaming error.

Matching model-info companions are provided under [`tests/fixtures`](tests/fixtures):

- `nanodet-m-320.onnx.modelinfo`
- `nanodet-plus-m-320.onnx.modelinfo`
- `nanodet-m-416.onnx.modelinfo`
- `nanodet-plus-m-416.onnx.modelinfo`

They describe Float32 outputs. For a model with a Float16 output, copy the
matching file and change the output `type` to `float16`. The supplied profiles
expect packed BGR input and include the required normalization ranges.

## Properties and metadata

| Property | Default | Description |
| --- | --- | --- |
| `tensor-id` | `nanodet-output` | Output tensor to decode |
| `label-file` | none | UTF-8 file containing exactly 80 non-empty labels, one per line |
| `score-threshold` | `0.3` | Minimum post-sigmoid class score |
| `iou-threshold` | `0.6` | Class-aware NMS threshold |
| `max-detections` | `100` | Maximum retained detections per frame (1-1000) |

These properties can be changed while the element is in READY or below.

Every retained result produces one object-detection entry and one related
classification entry. Without a label file, classes use stable names such as
`class-7`. Coordinates are clamped to the negotiated 320x320 or 416x416 frame.

When `videoscale add-borders=true` letterboxes the input, detections remain in
the model's coordinate space. The decoder does not remove the letterbox offset
when mapping detections back to the original frame.

## Example pipeline

Choose the model-info file matching the model family and input size. The decoder
infers NanoDet-m versus NanoDet-Plus from the negotiated output shape:

```sh
gst-launch-1.0 videotestsrc num-buffers=1 \
  ! videoconvert ! videoscale add-borders=true \
  ! video/x-raw,format=BGR,width=416,height=416 \
  ! tractinference model-file=/path/to/nanodet-plus-m-1.5x-416.onnx \
      model-info-file=analytics/nanodet/tests/fixtures/nanodet-plus-m-416.onnx.modelinfo \
  ! nanodettensordec tensor-id=nanodet-output \
  ! objectdetectionoverlay ! fakesink
```

Replace `tractinference` with `ortinference` to change runtimes without changing
the decoder configuration.
