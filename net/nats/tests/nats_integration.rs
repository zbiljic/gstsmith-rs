#![expect(
    clippy::expect_used,
    reason = "integration test setup and assertions require successful broker and GStreamer operations"
)]

use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use gst::prelude::*;

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstnats::plugin_register_static().expect("registering the NATS plugin");
    });
}

fn broker_url() -> String {
    std::env::var("NATS_TEST_URL").expect("NATS_TEST_URL must identify the integration-test broker")
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().expect("creating integration-test runtime")
}

fn unique_subject(label: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "gstsmith.test.{label}.{}.{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn connect_peer(runtime: &tokio::runtime::Runtime) -> async_nats::Client {
    let url = broker_url();
    runtime
        .block_on(async {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match async_nats::connect(url.clone()).await {
                    Ok(client) => return Ok(client),
                    Err(_error) if Instant::now() < deadline => {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        })
        .expect("connecting integration-test peer")
}

struct TestSubscriber<'a> {
    runtime: &'a tokio::runtime::Runtime,
    subscriber: Option<async_nats::Subscriber>,
}

impl<'a> TestSubscriber<'a> {
    fn subscribe(
        runtime: &'a tokio::runtime::Runtime,
        client: &async_nats::Client,
        subject: String,
    ) -> Self {
        let subscriber = runtime
            .block_on(client.subscribe(subject))
            .expect("creating peer subscription");
        Self {
            runtime,
            subscriber: Some(subscriber),
        }
    }

    fn next(&mut self, timeout_message: &str) -> async_nats::Message {
        self.next_with_timeout(Duration::from_secs(2), timeout_message)
    }

    fn next_with_timeout(
        &mut self,
        timeout: Duration,
        timeout_message: &str,
    ) -> async_nats::Message {
        let subscriber = self
            .subscriber
            .as_mut()
            .expect("test subscriber remains available");
        self.runtime
            .block_on(async { tokio::time::timeout(timeout, subscriber.next()).await })
            .expect(timeout_message)
            .expect("peer subscription remains open")
    }
}

impl Drop for TestSubscriber<'_> {
    fn drop(&mut self) {
        if let Some(mut subscriber) = self.subscriber.take() {
            self.runtime.block_on(async move {
                let _unsubscribe_result = subscriber.unsubscribe().await;
            });
        }
    }
}

fn readable_bytes(buffer: &gst::Buffer) -> Vec<u8> {
    buffer
        .map_readable()
        .expect("mapping buffer")
        .as_slice()
        .to_vec()
}

fn broker_socket_addr() -> std::net::SocketAddr {
    broker_url()
        .strip_prefix("nats://")
        .expect("NATS_TEST_URL uses the nats:// scheme")
        .parse()
        .expect("NATS_TEST_URL contains an IP socket address")
}

async fn proxy_connections(
    listener: tokio::net::TcpListener,
    target: std::net::SocketAddr,
) -> std::io::Result<()> {
    loop {
        let (mut incoming, _) = listener.accept().await?;
        let mut outgoing = tokio::net::TcpStream::connect(target).await?;
        tokio::spawn(async move {
            let _copy_result = tokio::io::copy_bidirectional(&mut incoming, &mut outgoing).await;
        });
    }
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn registration_pads_properties_and_initial_connection_policy() {
    init();
    assert!(gst::Registry::get().find_plugin("nats").is_some());
    for (factory, pad) in [("natssrc", "src"), ("natssink", "sink")] {
        let element = gst::ElementFactory::make(factory)
            .build()
            .expect("constructing NATS element");
        assert!(
            element
                .static_pad(pad)
                .expect("finding pad")
                .pad_template_caps()
                .is_any()
        );
        assert_eq!(
            element.property::<String>("servers"),
            "nats://127.0.0.1:4222"
        );
        assert_eq!(element.property::<u64>("connection-timeout"), 5_000_000_000);
        assert_eq!(element.property::<u32>("max-reconnects"), 0);
    }

    let unavailable = gst::ElementFactory::make("natssink")
        .property("servers", "nats://127.0.0.1:1")
        .property("subject", unique_subject("unavailable"))
        .property("connection-timeout", 50_000_000_u64)
        .build()
        .expect("constructing unavailable sink");
    assert_eq!(
        unavailable.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    unavailable
        .set_state(gst::State::Null)
        .expect("unavailable sink to NULL");

    let retrying = gst::ElementFactory::make("natssink")
        .property("servers", broker_url())
        .property("subject", unique_subject("retry"))
        .property("retry-on-initial-connect", true)
        .build()
        .expect("constructing retrying sink");
    assert_ne!(
        retrying.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    retrying
        .set_state(gst::State::Null)
        .expect("retrying sink to NULL");
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn natssink_publishes_binary_and_zero_length_messages() {
    init();
    let runtime = runtime();
    let peer = connect_peer(&runtime);
    let subject = unique_subject("sink-bytes");
    let mut subscriber = TestSubscriber::subscribe(&runtime, &peer, subject.clone());
    runtime
        .block_on(peer.flush())
        .expect("activating peer subscription");

    let sink = gst::ElementFactory::make("natssink")
        .property("servers", broker_url())
        .property("subject", &subject)
        .build()
        .expect("constructing sink");
    let mut harness = gst_check::Harness::with_element(&sink, Some("sink"), None);
    harness.set_src_caps_str("application/octet-stream");
    harness.play();
    harness
        .push(gst::Buffer::from_mut_slice(vec![0, 255, 17, 0]))
        .expect("pushing binary buffer");
    harness
        .push(gst::Buffer::from_mut_slice(Vec::<u8>::new()))
        .expect("pushing empty buffer");

    let first = subscriber.next("binary message timeout");
    let second = subscriber.next("empty message timeout");
    assert_eq!(first.payload.as_ref(), &[0, 255, 17, 0]);
    assert!(second.payload.is_empty());
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn natssrc_emits_caps_timestamps_and_complete_wildcard_envelope() {
    init();
    let runtime = runtime();
    let peer = connect_peer(&runtime);
    let stem = unique_subject("src");
    let wildcard = format!("{stem}.*");
    let actual = format!("{stem}.actual");
    let caps = gst::Caps::builder("application/x-nats-test").build();
    let source = gst::ElementFactory::make("natssrc")
        .property("servers", broker_url())
        .property("subject", &wildcard)
        .property("caps", &caps)
        .build()
        .expect("constructing source");
    let mut harness = gst_check::Harness::with_element(&source, None, Some("src"));
    harness.play();

    let mut headers = async_nats::HeaderMap::new();
    headers.append("X-Duplicate", "one");
    headers.append("X-Duplicate", "two");
    runtime
        .block_on(peer.publish_with_reply_and_headers(
            actual.clone(),
            "reply.target",
            headers,
            vec![7, 8, 9].into(),
        ))
        .expect("publishing peer message");
    runtime
        .block_on(peer.publish(actual.clone(), Vec::<u8>::new().into()))
        .expect("publishing empty peer message");
    runtime
        .block_on(peer.flush())
        .expect("flushing peer message");

    let buffer = harness.pull().expect("pulling source buffer");
    assert_eq!(readable_bytes(&buffer), vec![7, 8, 9]);
    assert!(buffer.pts().is_some());
    assert_eq!(
        source
            .static_pad("src")
            .expect("finding source pad")
            .current_caps(),
        Some(caps)
    );
    let meta = gst::meta::CustomMeta::from_buffer(&buffer, "GstNatsMessageMeta")
        .expect("source envelope metadata");
    assert_eq!(
        meta.structure()
            .get::<String>("subject")
            .expect("metadata subject"),
        actual
    );
    assert_eq!(
        meta.structure()
            .get::<String>("reply-subject")
            .expect("metadata reply"),
        "reply.target"
    );
    let metadata_headers = meta
        .structure()
        .get::<gst::Array>("headers")
        .expect("metadata headers");
    assert_eq!(metadata_headers.len(), 2);
    assert!(readable_bytes(&harness.pull().expect("pulling empty source buffer")).is_empty());
}

fn meta_buffer(payload: Vec<u8>, subject: &str, reply: &str) -> gst::Buffer {
    let mut buffer = gst::Buffer::from_mut_slice(payload);
    {
        let mut meta = gst::meta::CustomMeta::add(
            buffer.get_mut().expect("new buffer is writable"),
            "GstNatsMessageMeta",
        )
        .expect("adding NATS metadata");
        meta.mut_structure().set("subject", subject);
        meta.mut_structure().set("reply-subject", reply);
        meta.mut_structure().set(
            "headers",
            gst::Array::new([
                gst::Structure::builder("nats-header")
                    .field("name", "X-Duplicate")
                    .field("value", "one")
                    .build()
                    .to_send_value(),
                gst::Structure::builder("nats-header")
                    .field("name", "X-Duplicate")
                    .field("value", "two")
                    .build()
                    .to_send_value(),
            ]),
        );
    }
    buffer
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn natssink_republishes_envelope_and_fixed_subject_overrides_only_subject() {
    init();
    let runtime = runtime();
    let peer = connect_peer(&runtime);
    let meta_subject = unique_subject("meta");
    let fixed_subject = unique_subject("fixed");
    let mut meta_subscriber = TestSubscriber::subscribe(&runtime, &peer, meta_subject.clone());
    let mut fixed_subscriber = TestSubscriber::subscribe(&runtime, &peer, fixed_subject.clone());
    runtime
        .block_on(peer.flush())
        .expect("activating subscriptions");

    let dynamic_sink = gst::ElementFactory::make("natssink")
        .property("servers", broker_url())
        .build()
        .expect("constructing dynamic sink");
    let mut dynamic = gst_check::Harness::with_element(&dynamic_sink, Some("sink"), None);
    dynamic.set_src_caps_str("application/octet-stream");
    dynamic.play();
    dynamic
        .push(meta_buffer(vec![1], &meta_subject, "reply.dynamic"))
        .expect("publishing dynamic envelope");

    let dynamic_message = meta_subscriber.next("dynamic message timeout");
    assert_eq!(dynamic_message.reply.as_deref(), Some("reply.dynamic"));
    assert_eq!(
        dynamic_message
            .headers
            .expect("dynamic headers")
            .get_all("X-Duplicate")
            .count(),
        2
    );

    let fixed_sink = gst::ElementFactory::make("natssink")
        .property("servers", broker_url())
        .property("subject", &fixed_subject)
        .build()
        .expect("constructing fixed sink");
    let mut fixed = gst_check::Harness::with_element(&fixed_sink, Some("sink"), None);
    fixed.set_src_caps_str("application/octet-stream");
    fixed.play();
    fixed
        .push(meta_buffer(vec![2], &meta_subject, "reply.fixed"))
        .expect("publishing fixed envelope");
    let fixed_message = fixed_subscriber.next("fixed message timeout");
    assert_eq!(fixed_message.subject.as_str(), fixed_subject);
    assert_eq!(fixed_message.reply.as_deref(), Some("reply.fixed"));
    assert_eq!(
        fixed_message
            .headers
            .expect("fixed headers")
            .get_all("X-Duplicate")
            .count(),
        2
    );
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn blocked_source_and_sink_shutdown_are_bounded() {
    init();
    let source = gst::ElementFactory::make("natssrc")
        .property("servers", broker_url())
        .property("subject", unique_subject("blocked"))
        .build()
        .expect("constructing blocked source");
    source
        .set_state(gst::State::Playing)
        .expect("starting blocked source");
    let started = Instant::now();
    source
        .set_state(gst::State::Null)
        .expect("stopping blocked source");
    assert!(started.elapsed() < Duration::from_secs(2));

    let sink = gst::ElementFactory::make("natssink")
        .property("servers", broker_url())
        .property("subject", unique_subject("shutdown"))
        .property("drain-timeout", 0_u64)
        .build()
        .expect("constructing zero-drain sink");
    sink.set_state(gst::State::Playing)
        .expect("starting zero-drain sink");
    let started = Instant::now();
    sink.set_state(gst::State::Null)
        .expect("stopping zero-drain sink");
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn queue_group_sources_deliver_each_message_to_exactly_one_member() {
    init();
    let runtime = runtime();
    let peer = connect_peer(&runtime);
    let subject = unique_subject("queue");
    let queue_group = unique_subject("workers");
    let source_one = gst::ElementFactory::make("natssrc")
        .property("servers", broker_url())
        .property("subject", &subject)
        .property("queue-group", &queue_group)
        .build()
        .expect("constructing first queue source");
    let source_two = gst::ElementFactory::make("natssrc")
        .property("servers", broker_url())
        .property("subject", &subject)
        .property("queue-group", &queue_group)
        .build()
        .expect("constructing second queue source");
    let mut first = gst_check::Harness::with_element(&source_one, None, Some("src"));
    let mut second = gst_check::Harness::with_element(&source_two, None, Some("src"));
    first.play();
    second.play();

    runtime.block_on(async {
        for value in 0_u8..20 {
            peer.publish(subject.clone(), vec![value].into())
                .await
                .expect("publishing queue-group message");
        }
        peer.flush().await.expect("flushing queue-group messages");
    });

    let deadline = Instant::now() + Duration::from_secs(2);
    while first.buffers_received() + second.buffers_received() < 20 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(first.buffers_received() + second.buffers_received(), 20);

    let mut first_values = std::collections::BTreeSet::new();
    while let Some(buffer) = first.try_pull() {
        let bytes = readable_bytes(&buffer);
        first_values.insert(*bytes.first().expect("queue-group payload byte"));
    }
    let mut second_values = std::collections::BTreeSet::new();
    while let Some(buffer) = second.try_pull() {
        let bytes = readable_bytes(&buffer);
        second_values.insert(*bytes.first().expect("queue-group payload byte"));
    }
    assert!(!first_values.is_empty());
    assert!(!second_values.is_empty());
    assert!(first_values.is_disjoint(&second_values));
    first_values.extend(second_values);
    assert_eq!(first_values, (0_u8..20).collect());
}

#[test]
#[ignore = "requires NATS_TEST_URL and a real Core NATS server"]
fn retry_on_initial_connect_delivers_after_connectivity_appears() {
    init();
    let runtime = runtime();
    let peer = connect_peer(&runtime);
    let subject = unique_subject("late-connect");
    let mut subscriber = TestSubscriber::subscribe(&runtime, &peer, subject.clone());
    runtime
        .block_on(peer.flush())
        .expect("activating late-connect subscription");

    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("binding delayed proxy");
    let proxy_address = listener
        .local_addr()
        .expect("reading delayed proxy address");
    let sink = gst::ElementFactory::make("natssink")
        .property("servers", format!("nats://{proxy_address}"))
        .property("subject", &subject)
        .property("connection-timeout", 50_000_000_u64)
        .property("retry-on-initial-connect", true)
        .build()
        .expect("constructing retrying sink");
    let mut harness = gst_check::Harness::with_element(&sink, Some("sink"), None);
    harness.set_src_caps_str("application/octet-stream");
    harness.play();

    let proxy = runtime.spawn(proxy_connections(listener, broker_socket_addr()));
    harness
        .push(gst::Buffer::from_slice([42]))
        .expect("queueing message before delayed connection");
    let message =
        subscriber.next_with_timeout(Duration::from_secs(5), "late-connect message timeout");
    assert_eq!(message.payload.as_ref(), &[42]);

    drop(harness);
    proxy.abort();
    let _join_result = runtime.block_on(proxy);
}
