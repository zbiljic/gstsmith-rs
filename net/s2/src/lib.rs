//! `GStreamer` elements for S2 durable-stream byte transport.

use gst::glib;

mod config;
mod meta;
mod runtime;
mod s2sink;
mod s2src;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    meta::register();
    s2src::register(plugin)?;
    s2sink::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    s2,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
