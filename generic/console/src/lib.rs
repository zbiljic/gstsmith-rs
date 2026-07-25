//! `GStreamer` console byte transports and a text-oriented debug tap.

use gst::glib;

mod consoleprint;
mod consolesink;
mod consolesrc;
mod output;

pub use output::ConsoleStream;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    consolesrc::register(plugin)?;
    consoleprint::register(plugin)?;
    consolesink::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    console,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
