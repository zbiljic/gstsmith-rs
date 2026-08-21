//! Bounded process-wide `GStreamer` metrics exposed through `OpenMetrics`.

use gst::glib;

mod metrics;
mod prometheus;
mod server;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    prometheus::register(plugin)
}

gst::plugin_define!(
    prometheus,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
