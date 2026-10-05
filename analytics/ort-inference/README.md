# ORT inference

`gst-plugin-ort-inference` provides the `ortinference` GStreamer element. It
uses ONNX Runtime and publishes the model outputs model-info declares, in
model-info order, through the shared `tensor/strided` caps and `GstTensorMeta`
contract. Carrier buffers pass through unchanged. The shared
[model-info contract](../inference-common/README.md#model-info-contract) covers
output subsets and dimensions that bind the model's dynamic dimensions.

`input-mode=video` is the default. Set `input-mode=tensor-meta` to consume
one or more tensors prepared upstream in `GstTensorMeta`, with the
[shared tensor-input contract](../inference-common/README.md#tensor-input-inference).
The producer can be a preprocessor, a custom source, or another inference
element. Each model input selects a tensor by its model-info `id`, independently
of tensor order or the metadata instance containing it. The carrier buffer and
existing tensors are preserved; outputs are appended as a new `GstTensorMeta`.

Set `input-mode` in NULL or READY. Provider and session properties apply to
both modes; `model-channel-order` affects only video mode. Input mode is
explicit: existing tensor metadata never changes how video mode processes
pixels. Tensor mode does not read the carrier's payload as model input.

Tensor inputs and outputs may have non-unit leading dimensions. Batch-one
image rules apply only to video input. See the shared [model-info contract](../inference-common/README.md#model-info-contract)
for shape and layout requirements and [compatibility](../inference-common/README.md#compatibility)
for the scope of the fixture tests.

For primary/secondary model chaining, use two `ortinference` instances with
the secondary in `input-mode=tensor-meta`; map its input IDs to the primary's
output IDs. Use distinct output IDs and group IDs for each stage. See
[selection and per-object limitations](../inference-common/README.md#tensor-input-inference).

`model-channel-order=rgb` is the READY-mutable default. Set
`model-channel-order=bgr` for a model that expects BGR channel order. The
property affects tensor preprocessing only: truthful RGB/BGR/RGBA/BGRA caps
and video bytes pass through unchanged, and existing metadata is preserved.
Source pixel format does not imply model channel order. Three model-info
normalization ranges remain in semantic R, G, B order.

```sh
gst-launch-1.0 \
  ... \
  ! "video/x-raw,format=RGB,width=320,height=320" \
  ! ortinference \
      model-file=model.onnx \
      model-channel-order=bgr \
  ! ...
```

The `execution-provider` property defaults to `cpu`. The optional `coreml`
Cargo feature adds the `coreml` provider; requesting it when the selected ORT
runtime does not provide CoreML fails element startup rather than silently
falling back to CPU. When CoreML is available, unsupported graph nodes can
still run on ORT's CPU provider unless `strict-execution-provider=true`.
`intra-op-threads` is READY-mutable and zero leaves ORT's thread policy
unchanged. `graph-optimization` defaults to level 3.

The `coreml` feature also exposes the following properties. Set them in NULL
or READY; their values are applied when the inference session starts. They
apply only with `execution-provider=coreml`; CPU sessions ignore them.
Default values match ONNX Runtime's CoreML defaults.

| Property | Values | Default |
| --- | --- | --- |
| `coreml-model-format` | `neural-network`, `mlprogram` | `neural-network` |
| `coreml-compute-units` | `all`, `cpu-only`, `cpu-and-gpu`, `cpu-and-neural-engine` | `all` |
| `coreml-require-static-input-shapes` | Boolean | `false` |
| `coreml-model-cache-directory` | Directory path; unset or empty disables caching | Unset |
| `coreml-specialization-strategy` | `default`, `fast-prediction` | `default` |
| `coreml-profile-compute-plan` | Boolean | `false` |
| `coreml-enable-on-subgraphs` | Boolean | `false` |
| `coreml-allow-low-precision-accumulation-on-gpu` | Boolean | `false` |

`mlprogram` requires macOS 12+ or iOS 15+ and may improve operator coverage
and performance. Compute units select which devices CoreML may use; neither
`all` nor `cpu-and-neural-engine` guarantees Neural Engine execution.
Requiring static input shapes restricts which nodes CoreML accepts; it does
not specialize dynamic shapes or change the model-info contract. Nodes
CoreML cannot accept may run on ORT's CPU provider, or fail startup with
`strict-execution-provider=true`.

`fast-prediction` can trade model loading time, memory, and disk space for
prediction latency. Compute-plan profiling logs CoreML hardware placement
and estimated execution time; it is separate from ORT session profiling.
Enabling subgraphs allows CoreML inside `Loop`, `Scan`, and `If` bodies.
Low-precision GPU accumulation permits FP16 accumulation and can affect
numerical accuracy. Support for these options depends on the CoreML runtime;
see the [ORT CoreML option reference](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#available-options-new-api).

`coreml-model-cache-directory` enables reuse of compiled CoreML artifacts
across session starts. ORT creates the directory as needed. Its CoreML
provider derives the cache identity from the ONNX model's optional
`CACHE_KEY` entry in `metadata_props`, stored inside the `.onnx` file.
For a file-loaded model without that entry, ORT hashes the model path.
This is [ONNX Runtime's cache-key mechanism](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#available-options-new-api).

ORT does not automatically detect changed model contents or remove stale
cache entries. When updating a model, update its `CACHE_KEY` value if it has
one, use a new model path if it does not, or clear the old cache. Replacing
weights at the same path while retaining the same cache identity can reuse
stale artifacts. Cache invalidation is the caller's responsibility.

On macOS, run the following from the repository root to build with CoreML
and run the static convolution fixture with MLProgram and a persistent cache:

```sh
cargo build -p gst-plugin-ort-inference --features coreml
export GST_PLUGIN_PATH="$PWD/target/debug"
gst-launch-1.0 \
  videotestsrc num-buffers=1 \
  ! video/x-raw,format=RGB,width=2,height=2 \
  ! ortinference \
      model-file=analytics/tract-inference/tests/fixtures/metal-conv.onnx \
      execution-provider=coreml \
      coreml-model-format=mlprogram \
      coreml-compute-units=all \
      coreml-require-static-input-shapes=true \
      coreml-model-cache-directory="$PWD/target/coreml-cache" \
  ! fakesink
```

`strict-execution-provider` is a READY-mutable boolean that defaults to
`false`. When enabled with a non-CPU provider, it disables ONNX Runtime's CPU
execution-provider fallback, so startup fails unless that provider can own the
complete graph. It is invalid with `execution-provider=cpu`.

For example, a CoreML session can require complete ORT graph assignment with:

```sh
gst-launch-1.0 \
  ... \
  ! ortinference \
      model-file=model.onnx \
      execution-provider=coreml \
      strict-execution-provider=true \
  ! ...
```

Strict assignment is a diagnostic, not a benchmark or a guarantee that CoreML
will dispatch operations to a particular internal device. After ORT assigns a
graph partition to CoreML, CoreML may still choose the CPU, GPU, or Neural
Engine for its execution. Strict mode also does not imply zero-copy: it changes
provider assignment only and does not remove transfers between host memory and
the provider.

`Tensor::from_array` consumes the preprocessed `Vec` into an owned ORT value;
there is no additional Rust-side input copy. The session mutex serializes
access, and output bytes are copied into owned tensors before returning from
the streaming call. The Tract plugin can be built independently without
compiling this crate or installing an ORT runtime.

An ignored fixture benchmark is available for development diagnostics:

```sh
cargo test -p gst-plugin-ort-inference --test ortinference benchmark_fixture -- --ignored --nocapture
```

It measures development overhead, not production performance.

## Inspect a local model

The `model_info` example displays embedded model metadata and grouped lists
of inputs and outputs with their types and dimensions. Optionally supply
model-info and an input mode to run the element's CPU startup checks.

From the repository root:

```sh
# Inspect metadata.
cargo run -p gst-plugin-ort-inference --example model_info -- \
  analytics/inference-common/tests/fixtures/masked-sequence.onnx

# Check a prepared-tensor contract.
cargo run -p gst-plugin-ort-inference --example model_info -- \
  analytics/inference-common/tests/fixtures/tensor-axes.onnx \
  analytics/inference-common/tests/fixtures/tensor-axes.onnx.modelinfo \
  tensor-meta
```

Usage: `model_info MODEL.onnx [MODEL.modelinfo video|tensor-meta]`.
Use `video` to check the image input profile. Dimension names are
symbolic; `?` means an unnamed unknown dimension. These plugins currently need
fixed positive sizes in model-info; upstream `-1` wildcards are not yet
supported. Unsupported types and shapes are annotated.

With model-info, the example prints `Startup check: PASS` or exits nonzero
with the failure details. Without it, the contract is `NOT CHECKED`. PASS only
validates CPU startup: no buffers are sent, so inference, preprocessing,
decoding, and task correctness remain untested. The example does not download
models or generate model-info.
