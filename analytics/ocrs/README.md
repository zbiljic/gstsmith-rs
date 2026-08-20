# OCRs analysis

The `ocrs` plugin provides `ocrsanalysis`, an asynchronous local analyzer for
`video/x-raw,format=RGB`. It uses the OCRs engine and compatible detection and
recognition models. Video buffers pass through unchanged; source-correlated OCR
results are posted on the GStreamer bus.

## Example

Models are not bundled or downloaded. Supply compatible RTen models, such as
the files referenced by the [upstream OCRs project](https://github.com/robertknight/ocrs/tree/main/ocrs/examples).

```sh
gst-launch-1.0 \
  -m \
  videotestsrc num-buffers=20 is-live=true pattern=black \
  ! video/x-raw,width=640,height=240,framerate=2/1 \
  ! textoverlay text="HELLO OCR 123" \
      font-desc="Sans 48" \
      valignment=center \
      halignment=center \
  ! videoconvert \
  ! video/x-raw,format=RGB \
  ! ocrsanalysis \
      detection-model=/path/to/text-detection.rten \
      recognition-model=/path/to/text-recognition.rten \
  ! fakesink sync=true
```

Use `-m` to print the resulting element messages. Model compatibility and
licensing remain deployment responsibilities; the upstream pretrained-model
card is [CC-BY-SA-4.0](https://huggingface.co/robertknight/ocrs).

## Configuration

| Property | Default | Purpose |
|---|---:|---|
| `detection-model` | required | Compatible RTen text-detection model |
| `recognition-model` | required | Compatible RTen text-recognition model |
| `alphabet-file` | unset | Custom UTF-8 recognition alphabet |
| `allowed-characters` | unset | Restrict characters produced by recognition |
| `analysis-interval` | 500 ms | Minimum source-PTS interval; zero selects every buffer |
| `max-model-bytes` | 128 MiB | Per-model read limit; maximum 1 GiB |
| `max-frame-bytes` | 32 MiB | Packed RGB limit; maximum 128 MiB |
| `max-lines` | 128 | Lines retained per result; maximum 512 |
| `max-text-length` | 512 | Unicode scalars retained per line; maximum 1,024 |

The alphabet and allowlist are limited to 64 KiB, and an alphabet may contain
at most 4,095 Unicode scalars. Each model must expose one input and one output,
and required input-shape constraints are validated before the OCRs engine is
created. The engine validates output compatibility when it runs inference.

## Results and runtime behavior

`ocr-result` contains `request-id`, `generation`, optional `source-pts`, source
dimensions, latency, `full-text`, `line-count`, and an ordered `lines` array.
Each `ocr-line` contains text and a clamped, half-open `x`, `y`, `width`,
`height` box in stored-raster pixels. An image containing no text produces a
successful empty result.

`ocr-error` reports a sanitized, non-exhaustive `kind`; this plugin currently
emits `input` and `inference`. Consumers must accept unknown future kinds. No
confidence value is exposed because OCRs 0.12.2 does not provide recognition
confidence.

One worker owns the OCR engine and a capacity-one queue. If that queue is full,
the newest selected frame is dropped without blocking video streaming.
`submitted-frames`, `completed-frames`, `failed-frames`, and `dropped-frames`
report current-run totals. Results retain their request ID, source PTS,
dimensions, and generation so consumers can reject stale work. Stream-start,
segment, flush-stop, and backward-PTS events advance the generation.

Model graphs are trusted deployment configuration, not sandboxed input. Normal
shutdown may wait for active inference because RTen has no safe cancellation
API. `RTEN_NUM_THREADS` is process-wide rather than per element.

## Engine boundary

Other OCR engines belong in separate plugins such as `tesseractanalysis` or
`paddleocranalysis`, while reusing the `ocr-result` and `ocr-error` schema where
possible. Shared `ocr-common` code should be extracted only after another
implementation demonstrates real reuse.
