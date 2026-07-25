mod common;

use std::time::{Duration, Instant};

use gst::prelude::*;
use gst_base::prelude::*;

#[test]
fn source_is_live_time_formatted_and_supports_fixed_caps() {
    let source = common::element("s2src");
    let base_source = source
        .clone()
        .downcast::<gst_base::BaseSrc>()
        .expect("s2src is a base source");
    assert!(base_source.is_live());
    assert!(base_source.does_timestamp());
    let caps = gst::Caps::builder("application/x-s2-test").build();
    source.set_property("caps", &caps);
    assert_eq!(source.property::<Option<gst::Caps>>("caps"), Some(caps));
}

#[test]
fn source_rejects_missing_or_invalid_settings_before_networking() {
    for (basin, stream) in [
        (None, None),
        (Some("BAD"), Some("stream")),
        (Some("test-basin"), Some("")),
    ] {
        let source = common::element("s2src");
        if let Some(basin) = basin {
            source.set_property("basin", basin);
        }
        if let Some(stream) = stream {
            source.set_property("stream", stream);
        }
        let started = Instant::now();
        assert_eq!(
            source.set_state(gst::State::Paused),
            Err(gst::StateChangeError)
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        source
            .set_state(gst::State::Null)
            .expect("source returns to NULL");
    }
}

#[test]
fn source_start_controls_round_trip() {
    let source = common::element("s2src");
    source.set_property_from_str("start-mode", "sequence");
    source.set_property("start-seq-num", u64::MAX);
    source.set_property("start-timestamp", 17_u64);
    source.set_property("tail-offset", 9_u64);
    source.set_property("clamp-to-tail", true);
    source.set_property("ignore-command-records", true);
    assert_eq!(source.property::<u64>("start-seq-num"), u64::MAX);
    assert_eq!(source.property::<u64>("start-timestamp"), 17);
    assert_eq!(source.property::<u64>("tail-offset"), 9);
    assert!(source.property::<bool>("clamp-to-tail"));
    assert!(source.property::<bool>("ignore-command-records"));
}
