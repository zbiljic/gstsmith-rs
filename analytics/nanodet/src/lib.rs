//! NanoDet-m and NanoDet-Plus tensor decoding for `GStreamer` analytics pipelines.

use gst::glib;

mod nanodettensordec;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    nanodettensordec::register(plugin)
}

gst::plugin_define!(
    nanodet,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
