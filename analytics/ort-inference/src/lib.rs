//! Model-agnostic ONNX Runtime inference that publishes outputs as tensor metadata.

use gst::glib;

mod engine;
mod ortinference;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    ortinference::register(plugin)
}

gst::plugin_define!(
    ortinference,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
