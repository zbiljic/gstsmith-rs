# Inference common

`gst-inference-common` is an internal Rust library shared by gstsmith's
model-agnostic inference plugins. It owns the model-info 1.0 parser, image
preprocessing, engine-neutral tensor values, tensor caps construction, and
`GstTensorMeta` attachment.

It is an `rlib`, not a loadable GStreamer plugin. Backend crates link it
statically and remain independently installable. Public GStreamer factories
and backend-specific runtime configuration belong in those plugin crates.
