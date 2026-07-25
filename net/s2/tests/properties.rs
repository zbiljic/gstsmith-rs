mod common;

use gst::prelude::*;

fn assert_shared_defaults(element: &gst::Element) {
    assert_eq!(element.property::<Option<String>>("basin"), None);
    assert_eq!(element.property::<Option<String>>("stream"), None);
    assert_eq!(
        element.property::<Option<String>>("access-token-file"),
        None
    );
    assert_eq!(element.property::<Option<String>>("account-endpoint"), None);
    assert_eq!(element.property::<Option<String>>("basin-endpoint"), None);
    assert_eq!(element.property::<u64>("connection-timeout"), 3_000_000_000);
    assert_eq!(element.property::<u64>("request-timeout"), 5_000_000_000);
    assert_eq!(element.property::<u32>("retry-max-attempts"), 3);
    assert_eq!(element.property::<u64>("retry-min-delay"), 100_000_000);
    assert_eq!(element.property::<u64>("retry-max-delay"), 1_000_000_000);
    assert_eq!(element.property::<u32>("queue-capacity"), 64);
}

#[test]
fn registers_plugin_metadata_and_factories() {
    common::init();
    assert!(gst::Registry::get().find_plugin("s2").is_some());
    assert!(gst::ElementFactory::find("s2src").is_some());
    assert!(gst::ElementFactory::find("s2sink").is_some());
    assert!(gst::meta::CustomMeta::is_registered("GstS2RecordMeta"));
}

#[test]
fn property_defaults_match_contract() {
    let source = common::element("s2src");
    assert_shared_defaults(&source);
    assert_eq!(source.property::<Option<gst::Caps>>("caps"), None);
    assert_eq!(source.property::<u64>("start-seq-num"), 0);
    assert_eq!(source.property::<u64>("start-timestamp"), 0);
    assert_eq!(source.property::<u64>("tail-offset"), 0);
    assert!(!source.property::<bool>("clamp-to-tail"));
    assert!(!source.property::<bool>("ignore-command-records"));

    let sink = common::element("s2sink");
    assert_shared_defaults(&sink);
    assert_eq!(sink.property::<u64>("batch-linger"), 5_000_000);
    assert_eq!(sink.property::<u32>("batch-max-records"), 1_000);
    assert_eq!(sink.property::<u32>("batch-max-bytes"), 1_048_576);
    assert_eq!(sink.property::<u32>("max-unacked-bytes"), 5_242_880);
    assert_eq!(sink.property::<Option<String>>("fencing-token-file"), None);
    assert!(!sink.property::<bool>("match-seq-num-enabled"));
    assert_eq!(sink.property::<u64>("match-seq-num"), 0);
    assert!(!sink.property::<bool>("preserve-timestamp"));
    assert_eq!(sink.property::<u64>("shutdown-timeout"), 10_000_000_000);
}

#[test]
fn properties_round_trip_and_are_ready_mutable() {
    for factory in ["s2src", "s2sink"] {
        let element = common::element(factory);
        element.set_property("basin", "test-basin");
        element.set_property("stream", "test-stream");
        element.set_property("queue-capacity", 7_u32);
        element.set_property("connection-timeout", 17_u64);
        assert_eq!(
            element.property::<Option<String>>("basin").as_deref(),
            Some("test-basin")
        );
        assert_eq!(
            element.property::<Option<String>>("stream").as_deref(),
            Some("test-stream")
        );
        assert_eq!(element.property::<u32>("queue-capacity"), 7);
        assert_eq!(element.property::<u64>("connection-timeout"), 17);
        let shared = [
            "basin",
            "stream",
            "access-token-file",
            "account-endpoint",
            "basin-endpoint",
            "connection-timeout",
            "request-timeout",
            "retry-max-attempts",
            "retry-min-delay",
            "retry-max-delay",
            "compression",
            "queue-capacity",
        ];
        let source = [
            "caps",
            "start-mode",
            "start-seq-num",
            "start-timestamp",
            "tail-offset",
            "clamp-to-tail",
            "ignore-command-records",
        ];
        let sink = [
            "batch-linger",
            "batch-max-records",
            "batch-max-bytes",
            "max-unacked-bytes",
            "append-retry-policy",
            "fencing-token-file",
            "match-seq-num-enabled",
            "match-seq-num",
            "preserve-timestamp",
            "shutdown-timeout",
        ];
        let specific = if factory == "s2src" {
            source.as_slice()
        } else {
            sink.as_slice()
        };
        for name in shared.iter().chain(specific) {
            let property = element.find_property(name).expect("finding S2 property");
            assert!(
                property.flags().contains(gst::PARAM_FLAG_MUTABLE_READY),
                "{factory}:{} must be mutable through READY",
                property.name()
            );
        }
    }
}

#[test]
fn no_raw_secret_or_insecure_tls_property_is_exposed() {
    for factory in ["s2src", "s2sink"] {
        let element = common::element(factory);
        let names = element
            .list_properties()
            .iter()
            .map(gst::glib::ParamSpec::name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"access-token-file"));
        assert!(!names.contains(&"access-token"));
        assert!(!names.contains(&"insecure-skip-cert-verification"));
        if factory == "s2sink" {
            assert!(names.contains(&"fencing-token-file"));
            assert!(!names.contains(&"fencing-token"));
        }
    }
}

#[test]
fn pads_advertise_any_caps() {
    for (factory, pad) in [("s2src", "src"), ("s2sink", "sink")] {
        assert!(
            common::element(factory)
                .static_pad(pad)
                .expect("finding static pad")
                .pad_template_caps()
                .is_any()
        );
    }
}
