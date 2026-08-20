//! Local, frame-correlated OCRs analysis for RGB video streams.

use gst::glib;

mod backend;
mod message;
mod ocrsanalysis;
mod worker;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    ocrsanalysis::register(plugin)
}

gst::plugin_define!(
    ocrs,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
