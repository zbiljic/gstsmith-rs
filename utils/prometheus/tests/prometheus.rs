use std::sync::Once;
use std::{path::Path, process::Command};

use gst::prelude::*;

#[expect(
    clippy::expect_used,
    reason = "all tests require successful one-time GStreamer and plugin initialization"
)]
fn init() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstprometheus::plugin_register_static().expect("registering the Prometheus plugin");
    });
}

#[test]
fn registers_prometheus_tracer() {
    init();

    assert!(
        gst::TracerFactory::factories()
            .iter()
            .any(|factory| factory.name() == "prometheus")
    );
}

#[test]
fn gst_tracers_environment_accepts_startup_properties() {
    let executable = std::env::current_exe().expect("locating integration-test executable");
    let plugin_path = executable
        .parent()
        .and_then(Path::parent)
        .expect("finding Cargo target profile directory");
    let output = Command::new("gst-launch-1.0")
        .env("GST_PLUGIN_PATH", plugin_path)
        .env(
            "GST_TRACERS",
            r#"prometheus(listen="127.0.0.1:0",exclude-filter=".*",max-pad-series=(uint)7)"#,
        )
        .args([
            "-q",
            "fakesrc",
            "num-buffers=1",
            "!",
            "fakesink",
            "sync=false",
        ])
        .output()
        .expect("running parameterized tracer pipeline");

    assert!(
        output.status.success(),
        "parameterized tracer pipeline failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("Can't setup tracer"),
        "parameterized tracer was not configured: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
