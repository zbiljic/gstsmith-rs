# Inference common

`gst-inference-common` is an internal Rust library shared by gstsmith's
model-agnostic inference plugins. It owns the model-info 1.0 parser, image
preprocessing, engine-neutral tensor values, tensor caps construction, and
`GstTensorMeta` attachment.

Its deterministic ONNX/model-info fixture is also the shared compatibility
contract used by backend parity tests.

Preprocessing decodes truthful RGB, BGR, RGBA, or BGRA source pixels into
semantic red, green, and blue values, then packs them in the channel order
requested by the backend. The default model order is RGB; BGR is an explicit
opt-in. When model-info declares three normalization ranges, their order is
always semantic R, G, B regardless of the model's channel order.
Backend elements expose this choice as `model-channel-order`; for example,
configure `tractinference model-channel-order=bgr` for BGR channel order.

It is an `rlib`, not a loadable GStreamer plugin. Backend crates link it
statically and remain independently installable. Public GStreamer factories
and backend-specific runtime configuration belong in those plugin crates.
