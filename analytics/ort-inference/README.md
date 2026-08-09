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
