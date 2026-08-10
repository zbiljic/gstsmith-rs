# ORT inference

`gst-plugin-ort-inference` provides the `ortinference` GStreamer element. It
uses ONNX Runtime and publishes every model output in model-info order through
the shared `tensor/strided` caps and `GstTensorMeta` contract. Video buffers
pass through unchanged.

The `execution-provider` property defaults to `cpu`. The optional `coreml`
Cargo feature adds the `coreml` provider; requesting it when the selected ORT
runtime does not provide CoreML fails element startup rather than silently
falling back to CPU. `intra-op-threads` is READY-mutable and zero leaves ORT's
thread policy unchanged. `graph-optimization` defaults to level 3.

`strict-execution-provider` is a READY-mutable boolean that defaults to
`false`. When enabled with a non-CPU provider, it disables ONNX Runtime's CPU
execution-provider fallback, so startup fails unless that provider can own the
complete graph. It is invalid with `execution-provider=cpu`.

For example, a CoreML session can require complete ORT graph assignment with:

```sh
gst-launch-1.0 ... ! ortinference model-file=model.onnx \
  execution-provider=coreml strict-execution-provider=true ! ...
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
the streaming call. The plugin is independent of the Tract backend; a
Tract-only build does not resolve this crate or the ORT runtime.

An ignored fixture benchmark is available for development diagnostics:

```sh
cargo test -p gst-plugin-ort-inference --test ortinference benchmark_fixture -- --ignored --nocapture
```

It measures development overhead, not production performance.
