//! Asynchronous provider-neutral vision-language analysis for JPEG streams.

use gst::glib;

mod backend;
mod prompt;
mod runtime;
mod vlmanalysis;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    vlmanalysis::register(plugin)
}

gst::plugin_define!(
    vlm,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
