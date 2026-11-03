//! Model-agnostic ONNX Runtime inference that publishes outputs as tensor metadata.

use std::sync::OnceLock;

use gst::glib;

#[cfg(feature = "coreml")]
mod coreml;
mod engine;
mod ortinference;

/// Shares the one ONNX Runtime environment allowed per process with plugin callers.
pub fn environment() -> Result<&'static ort::environment::Environment, String> {
    static ENVIRONMENT: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| {
            ort::init()
                .build()
                .map_err(|error| format!("failed to create ONNX Runtime environment: {error}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

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
