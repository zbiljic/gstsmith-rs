#![expect(
    clippy::expect_used,
    reason = "test setup and assertions require successful GStreamer operations"
)]

use std::sync::Once;

use gst::prelude::*;

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstnats::plugin_register_static().expect("registering the NATS plugin");
    });
}

fn element(factory: &str) -> gst::Element {
    init();
    gst::ElementFactory::make(factory)
        .build()
        .expect("constructing NATS element")
}

#[test]
fn nats_registers_plugin_meta_and_factories() {
    init();
    assert!(gst::Registry::get().find_plugin("nats").is_some());
    assert!(gst::ElementFactory::find("natssrc").is_some());
    assert!(gst::ElementFactory::find("natssink").is_some());
    assert!(gst::meta::CustomMeta::is_registered("GstNatsMessageMeta"));
}

#[test]
fn nats_pads_advertise_any_caps() {
    let source = element("natssrc");
    assert!(
        source
            .static_pad("src")
            .expect("finding source pad")
            .pad_template_caps()
            .is_any()
    );
    let sink = element("natssink");
    assert!(
        sink.static_pad("sink")
            .expect("finding sink pad")
            .pad_template_caps()
            .is_any()
    );
}

fn assert_shared_defaults(element: &gst::Element) {
    assert_eq!(
        element.property::<String>("servers"),
        "nats://127.0.0.1:4222"
    );
    assert_eq!(element.property::<Option<String>>("connection-name"), None);
    assert_eq!(element.property::<Option<String>>("credentials-file"), None);
    assert_eq!(element.property::<Option<String>>("nkey-file"), None);
    assert!(!element.property::<bool>("tls-required"));
    assert_eq!(element.property::<Option<String>>("tls-ca-file"), None);
    assert_eq!(
        element.property::<Option<String>>("tls-client-cert-file"),
        None
    );
    assert_eq!(
        element.property::<Option<String>>("tls-client-key-file"),
        None
    );
    assert_eq!(element.property::<u64>("connection-timeout"), 5_000_000_000);
    assert_eq!(element.property::<u32>("max-reconnects"), 0);
    assert!(!element.property::<bool>("retry-on-initial-connect"));
}

#[test]
fn nats_property_defaults_match_contract() {
    let source = element("natssrc");
    assert_shared_defaults(&source);
    assert_eq!(source.property::<String>("subject"), "");
    assert_eq!(source.property::<String>("queue-group"), "");
    assert_eq!(source.property::<u32>("subscription-capacity"), 1024);
    assert_eq!(source.property::<Option<gst::Caps>>("caps"), None);

    let sink = element("natssink");
    assert_shared_defaults(&sink);
    assert_eq!(sink.property::<String>("subject"), "");
    assert_eq!(sink.property::<u32>("queue-capacity"), 64);
    assert!(!sink.property::<bool>("drop-on-full"));
    assert_eq!(sink.property::<u64>("drain-timeout"), 2_000_000_000);
    assert_eq!(sink.property::<u64>("dropped-messages"), 0);
}

#[test]
fn nats_properties_round_trip_in_ready() {
    let source = element("natssrc");
    source.set_property("servers", "nats://localhost:4223");
    source.set_property("connection-name", "source-test");
    source.set_property("subject", "events.>");
    source.set_property("queue-group", "workers");
    source.set_property("subscription-capacity", 7_u32);
    let caps = gst::Caps::builder("application/x-nats-test").build();
    source.set_property("caps", &caps);
    assert_eq!(
        source.property::<String>("servers"),
        "nats://localhost:4223"
    );
    assert_eq!(
        source
            .property::<Option<String>>("connection-name")
            .as_deref(),
        Some("source-test")
    );
    assert_eq!(source.property::<String>("subject"), "events.>");
    assert_eq!(source.property::<String>("queue-group"), "workers");
    assert_eq!(source.property::<u32>("subscription-capacity"), 7);
    assert_eq!(source.property::<Option<gst::Caps>>("caps"), Some(caps));

    let sink = element("natssink");
    sink.set_property("subject", "events.out");
    sink.set_property("queue-capacity", 3_u32);
    sink.set_property("drop-on-full", true);
    sink.set_property("drain-timeout", 17_u64);
    assert_eq!(sink.property::<String>("subject"), "events.out");
    assert_eq!(sink.property::<u32>("queue-capacity"), 3);
    assert!(sink.property::<bool>("drop-on-full"));
    assert_eq!(sink.property::<u64>("drain-timeout"), 17);
}

#[test]
fn nats_properties_have_ready_mutability_and_counter_is_read_only() {
    let shared = [
        "servers",
        "connection-name",
        "credentials-file",
        "nkey-file",
        "tls-required",
        "tls-ca-file",
        "tls-client-cert-file",
        "tls-client-key-file",
        "connection-timeout",
        "max-reconnects",
        "retry-on-initial-connect",
    ];
    for (factory, specific) in [
        (
            "natssrc",
            &["subject", "queue-group", "subscription-capacity", "caps"][..],
        ),
        (
            "natssink",
            &["subject", "queue-capacity", "drop-on-full", "drain-timeout"][..],
        ),
    ] {
        let element = element(factory);
        for property in shared.iter().chain(specific.iter()) {
            let pspec = element.find_property(property).expect("finding property");
            assert!(
                pspec.flags().contains(gst::PARAM_FLAG_MUTABLE_READY),
                "{factory}:{} must be mutable through READY",
                pspec.name()
            );
        }
    }
    let counter = element("natssink")
        .find_property("dropped-messages")
        .expect("finding dropped-messages");
    assert!(counter.flags().contains(gst::glib::ParamFlags::READABLE));
    assert!(!counter.flags().contains(gst::glib::ParamFlags::WRITABLE));
}

#[test]
fn natssrc_rejects_missing_subject_before_network_setup() {
    let source = element("natssrc");
    assert_eq!(
        source.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    source
        .set_state(gst::State::Null)
        .expect("source back to NULL");
}

#[test]
fn nats_rejects_malformed_shared_settings_before_network_setup() {
    for factory in ["natssrc", "natssink"] {
        let element = element(factory);
        element.set_property("servers", " , ");
        element.set_property("subject", "events");
        assert_eq!(
            element.set_state(gst::State::Paused),
            Err(gst::StateChangeError)
        );
        element
            .set_state(gst::State::Null)
            .expect("element back to NULL");
    }
}

#[test]
fn nats_custom_meta_preserves_duplicate_headers_on_copy() {
    init();
    let mut buffer = gst::Buffer::new();
    let mut meta = gst::meta::CustomMeta::add(
        buffer.get_mut().expect("new buffer is writable"),
        "GstNatsMessageMeta",
    )
    .expect("adding metadata");
    meta.mut_structure().set("subject", "events.actual");
    meta.mut_structure().set("reply-subject", "events.reply");
    meta.mut_structure().set(
        "headers",
        gst::Array::new([
            gst::Structure::builder("nats-header")
                .field("name", "X-Test")
                .field("value", "one")
                .build()
                .to_send_value(),
            gst::Structure::builder("nats-header")
                .field("name", "X-Test")
                .field("value", "two")
                .build()
                .to_send_value(),
        ]),
    );

    let copied = buffer.copy();
    let copied_meta = gst::meta::CustomMeta::from_buffer(&copied, "GstNatsMessageMeta")
        .expect("metadata survived copy");
    let headers = copied_meta
        .structure()
        .get::<gst::Array>("headers")
        .expect("headers are an array");
    assert_eq!(headers.len(), 2);
}
