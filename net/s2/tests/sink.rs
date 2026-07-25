mod common;

use std::time::{Duration, Instant};

use gst::prelude::*;

#[test]
fn sink_rejects_invalid_settings_before_networking() {
    let sink = common::element("s2sink");
    sink.set_property("basin", "test-basin");
    sink.set_property("stream", "test-stream");
    sink.set_property("account-endpoint", "http://user@example.test");
    sink.set_property("basin-endpoint", "http://example.test");
    let started = Instant::now();
    assert_eq!(
        sink.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    sink.set_state(gst::State::Null)
        .expect("sink returns to NULL");
}

#[test]
fn sink_append_controls_round_trip() {
    let sink = common::element("s2sink");
    sink.set_property("batch-linger", 0_u64);
    sink.set_property("batch-max-records", 3_u32);
    sink.set_property("batch-max-bytes", 128_u32);
    sink.set_property("max-unacked-bytes", 1_048_576_u32);
    sink.set_property_from_str("append-retry-policy", "all");
    sink.set_property("match-seq-num-enabled", true);
    sink.set_property("match-seq-num", 42_u64);
    sink.set_property("preserve-timestamp", true);
    sink.set_property("shutdown-timeout", 17_u64);
    assert_eq!(sink.property::<u64>("batch-linger"), 0);
    assert_eq!(sink.property::<u32>("batch-max-records"), 3);
    assert_eq!(sink.property::<u32>("batch-max-bytes"), 128);
    assert_eq!(sink.property::<u32>("max-unacked-bytes"), 1_048_576);
    assert!(sink.property::<bool>("match-seq-num-enabled"));
    assert_eq!(sink.property::<u64>("match-seq-num"), 42);
    assert!(sink.property::<bool>("preserve-timestamp"));
    assert_eq!(sink.property::<u64>("shutdown-timeout"), 17);
}

#[test]
fn sink_batch_property_bounds_match_service_contract() {
    let sink = common::element("s2sink");
    let records = sink
        .find_property("batch-max-records")
        .expect("batch-max-records property")
        .downcast::<gst::glib::ParamSpecUInt>()
        .expect("unsigned property");
    assert_eq!(records.minimum(), 1);
    assert_eq!(records.maximum(), 1_000);
    let bytes = sink
        .find_property("batch-max-bytes")
        .expect("batch-max-bytes property")
        .downcast::<gst::glib::ParamSpecUInt>()
        .expect("unsigned property");
    assert_eq!(bytes.minimum(), 8);
    assert_eq!(bytes.maximum(), 1_048_576);
}
