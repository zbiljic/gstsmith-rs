//! Bounded process-wide `GStreamer` metrics pushed with DogStatsD-compatible tags.

use gst::glib;

mod metrics;
pub mod statsd;
mod worker;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    statsd::register(plugin)
}

gst::plugin_define!(
    statsd,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
