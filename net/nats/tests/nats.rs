#![expect(
    clippy::expect_used,
    reason = "test setup and assertions require successful GStreamer operations"
)]

use std::sync::Once;
use std::time::Duration;

use gst::prelude::*;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};

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
    assert!(sink.property::<gst::Array>("headers").is_empty());
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
    let headers = gst::Array::new([
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
    ]);
    sink.set_property("subject", "events.out");
    sink.set_property("headers", &headers);
    sink.set_property("queue-capacity", 3_u32);
    sink.set_property("drop-on-full", true);
    sink.set_property("drain-timeout", 17_u64);
    assert_eq!(sink.property::<String>("subject"), "events.out");
    assert_eq!(sink.property::<gst::Array>("headers").len(), 2);
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
            &[
                "subject",
                "headers",
                "queue-capacity",
                "drop-on-full",
                "drain-timeout",
            ][..],
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
fn natssink_rejects_malformed_fixed_headers_before_network_setup() {
    let sink = element("natssink");
    sink.set_property("subject", "events");
    sink.set_property(
        "headers",
        gst::Array::new([gst::Structure::builder("not-a-nats-header")
            .field("name", "X-Test")
            .field("value", "one")
            .build()
            .to_send_value()]),
    );
    assert_eq!(
        sink.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    sink.set_state(gst::State::Null).expect("sink back to NULL");
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

fn permission_denying_server(
    runtime: &tokio::runtime::Runtime,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("binding local NATS protocol stub");
    let address = listener.local_addr().expect("stub address");
    let server = runtime.spawn(async move {
        let (stream, _) = listener.accept().await.expect("accepting NATS client");
        let (reader, mut writer) = stream.into_split();
        writer
            .write_all(b"INFO {\"server_id\":\"test\",\"max_payload\":1048576}\r\n")
            .await
            .expect("sending server info");
        let mut reader = tokio::io::BufReader::new(reader);
        let mut line = String::new();
        while reader.read_line(&mut line).await.expect("reading command") != 0 {
            let response = if line.starts_with("PING") {
                &b"PONG\r\n"[..]
            } else if line.starts_with("SUB ") {
                &b"-ERR 'Permissions Violation for Subscription to \"private.rejection.test\"'\r\n"
                    [..]
            } else if line.starts_with("PUB ") {
                let size = line
                    .split_ascii_whitespace()
                    .next_back()
                    .expect("publish size")
                    .parse::<usize>()
                    .expect("numeric publish size");
                assert!(size <= 1024, "only small test payloads are expected");
                let mut payload = vec![0; size + 2];
                reader
                    .read_exact(&mut payload)
                    .await
                    .expect("reading payload");
                &b"-ERR 'Permissions Violation for Publish to \"private.rejection.test\"'\r\n"[..]
            } else {
                &b""[..]
            };
            writer.write_all(response).await.expect("sending response");
            line.clear();
        }
    });
    (format!("nats://{address}"), server)
}

#[test]
fn permission_denials_post_sanitized_errors_and_reset_on_restart() {
    let runtime = tokio::runtime::Runtime::new().expect("test server runtime");
    for factory in ["natssrc", "natssink"] {
        let element = element(factory);
        element.set_property("subject", "private.rejection.test");
        let bus = gst::Bus::new();
        element.set_bus(Some(&bus));
        for _run in 0..2 {
            let (url, server) = permission_denying_server(&runtime);
            element.set_property("servers", &url);
            let mut harness = if factory == "natssink" {
                let mut harness = gst_check::Harness::with_element(&element, Some("sink"), None);
                harness.set_src_caps_str("application/octet-stream");
                harness.play();
                assert_eq!(
                    harness.push(gst::Buffer::from_slice(b"test")),
                    Ok(gst::FlowSuccess::Ok)
                );
                harness
            } else {
                let mut harness = gst_check::Harness::with_element(&element, None, Some("src"));
                harness.play();
                harness
            };
            let message = bus
                .iter_timed_filtered(gst::ClockTime::from_seconds(2), &[gst::MessageType::Error])
                .find(|message| {
                    matches!(message.view(), gst::MessageView::Error(error)
                        if error.error().matches(gst::ResourceError::NotAuthorized))
                })
                .expect("broker rejection must reach the bus");
            let gst::MessageView::Error(error) = message.view() else {
                panic!("expected an error message");
            };
            assert!(error.error().matches(gst::ResourceError::NotAuthorized));
            let debug = error.debug().expect("sanitized error detail");
            assert!(debug.ends_with("Core NATS server denied permission for an operation"));
            assert!(!debug.contains("private.rejection.test"));
            assert!(!error.error().message().contains("private.rejection.test"));
            if factory == "natssink" {
                assert_eq!(
                    harness.push(gst::Buffer::from_slice(b"after rejection")),
                    Err(gst::FlowError::Error)
                );
            } else {
                assert_eq!(harness.buffers_in_queue(), 0);
            }
            drop(harness);
            element
                .set_state(gst::State::Null)
                .expect("stopping rejected element");
            runtime
                .block_on(async { tokio::time::timeout(Duration::from_secs(2), server).await })
                .expect("rejected connection must close")
                .expect("protocol stub completed without panic");
            while bus.pop().is_some() {}
        }
    }
}
