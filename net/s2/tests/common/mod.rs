#![expect(
    clippy::expect_used,
    reason = "integration test setup requires successful GStreamer operations"
)]

use std::sync::Once;

pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gsts2::plugin_register_static().expect("registering the S2 plugin");
    });
}

pub fn element(factory: &str) -> gst::Element {
    init();
    gst::ElementFactory::make(factory)
        .build()
        .expect("constructing S2 element")
}
