//! `GStreamer` elements for Core NATS byte-message transport.

use gst::glib;

mod connection;
mod message_meta;
mod natssink;
mod natssrc;
mod runtime;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    message_meta::register();
    natssrc::register(plugin)?;
    natssink::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    nats,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
