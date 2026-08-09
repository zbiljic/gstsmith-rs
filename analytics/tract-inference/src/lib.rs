//! Model-agnostic Tract ONNX inference that publishes outputs as tensor metadata.

use gst::glib;

mod engine;
mod tractinference;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    tractinference::register(plugin)
}

gst::plugin_define!(
    tractinference,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
