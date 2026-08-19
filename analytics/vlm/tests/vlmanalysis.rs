#![expect(
    clippy::expect_used,
    reason = "test fixtures abort immediately when their local-only setup fails"
)]

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Once, mpsc};
use std::time::{Duration, Instant};

use base64::Engine as _;
use bytes::Bytes;
use gst::prelude::*;
use http_body_util::{BodyExt, Full};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;

#[derive(Clone)]
struct Reply {
    status: StatusCode,
    body: Vec<u8>,
    delay: Duration,
    location: Option<String>,
}

impl Reply {
    fn json(body: &str) -> Self {
        Self {
            status: StatusCode::OK,
            body: body.as_bytes().to_vec(),
            delay: Duration::ZERO,
            location: None,
        }
    }
}

struct TestServer {
    endpoint: String,
    requests: mpsc::Receiver<RecordedRequest>,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct RecordedRequest {
    method: hyper::Method,
    uri: String,
    headers: hyper::HeaderMap,
    body: Vec<u8>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("joining bounded test server");
        }
    }
}

fn make_server(replies: Vec<Reply>) -> TestServer {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("binding loopback test server");
    listener
        .set_nonblocking(true)
        .expect("making loopback listener nonblocking");
    let address = listener.local_addr().expect("reading loopback address");
    let (request_sender, requests) = mpsc::channel();
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_shutdown = Arc::clone(&shutdown);
    let replies = Arc::new(replies);
    let next = Arc::new(AtomicUsize::new(0));
    let thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("building local server runtime");
        while !thread_shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _peer)) => {
                    stream
                        .set_nonblocking(true)
                        .expect("making accepted test stream nonblocking");
                    let stream = {
                        let _guard = runtime.enter();
                        tokio::net::TcpStream::from_std(stream)
                            .expect("adopting accepted test stream")
                    };
                    let replies = Arc::clone(&replies);
                    let next = Arc::clone(&next);
                    let request_sender = request_sender.clone();
                    runtime.spawn(async move {
                        let service = service_fn(move |request: Request<hyper::body::Incoming>| {
                            let replies = Arc::clone(&replies);
                            let next = Arc::clone(&next);
                            let request_sender = request_sender.clone();
                            async move {
                                let method = request.method().clone();
                                let uri = request.uri().to_string();
                                let headers = request.headers().clone();
                                let body = request
                                    .into_body()
                                    .collect()
                                    .await
                                    .expect("collecting bounded test request")
                                    .to_bytes()
                                    .to_vec();
                                request_sender
                                    .send(RecordedRequest {
                                        method,
                                        uri,
                                        headers,
                                        body,
                                    })
                                    .expect("recording test request");
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                let reply = replies
                                    .get(index)
                                    .or_else(|| replies.last())
                                    .cloned()
                                    .expect("test server has a reply");
                                tokio::time::sleep(reply.delay).await;
                                let mut response =
                                    Response::new(Full::new(Bytes::from(reply.body)));
                                *response.status_mut() = reply.status;
                                if let Some(location) = reply.location {
                                    response.headers_mut().insert(
                                        hyper::header::LOCATION,
                                        location.parse().expect("valid redirect location"),
                                    );
                                }
                                Ok::<_, Infallible>(response)
                            }
                        });
                        let _connection = hyper::server::conn::http1::Builder::new()
                            .serve_connection(TokioIo::new(stream), service)
                            .await;
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_error) => break,
            }
            runtime.block_on(async {
                tokio::time::sleep(Duration::from_millis(5)).await;
            });
        }
        runtime.shutdown_timeout(Duration::from_millis(100));
    });
    TestServer {
        endpoint: format!("http://{address}/v1/chat/completions"),
        requests,
        shutdown,
        thread: Some(thread),
    }
}

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstvlm::plugin_register_static().expect("registering VLM plugin");
    });
}

fn make_element(endpoint: &str) -> gst::Element {
    init();
    gst::ElementFactory::make("vlmanalysis")
        .property("endpoint", endpoint)
        .property("model", "test-model")
        .property("analysis-interval", 0_u64)
        .build()
        .expect("constructing vlmanalysis")
}

fn make_harness(element: &gst::Element) -> gst_check::Harness {
    let mut harness = gst_check::Harness::with_element(element, Some("sink"), Some("src"));
    harness.set_src_caps_str("image/jpeg");
    harness.play();
    harness
}

fn jpeg(bytes: &[u8], pts: Option<gst::ClockTime>) -> gst::Buffer {
    let mut buffer = gst::Buffer::from_mut_slice(bytes.to_vec());
    buffer
        .get_mut()
        .expect("writable JPEG fixture")
        .set_pts(pts);
    buffer
}

fn ok_reply() -> Reply {
    Reply::json(
        r#"{"choices":[{"message":{"content":"description"}}],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#,
    )
}

fn wait_for_counter(element: &gst::Element, property: &str, expected: u64) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if element.property::<u64>(property) >= expected {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(element.property::<u64>(property), expected);
}

fn wait_for_outcomes(element: &gst::Element, expected: u64) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let outcomes = element.property::<u64>("completed-requests")
            + element.property::<u64>("failed-requests");
        if outcomes >= expected {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        element.property::<u64>("completed-requests") + element.property::<u64>("failed-requests"),
        expected
    );
}

fn wait_for_structure(bus: &gst::Bus, name: &str) -> Option<gst::Structure> {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let Some(message) = bus.timed_pop_filtered(
            gst::ClockTime::from_mseconds(100),
            &[gst::MessageType::Element],
        ) else {
            continue;
        };
        if let Some(structure) = message.structure()
            && structure.name() == name
        {
            return Some(structure.to_owned());
        }
    }
    None
}

#[test]
fn passthrough_preserves_payload_timestamps_offsets_flags_and_meta() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    let mut harness = make_harness(&element);
    let mut input = jpeg(b"jpeg-payload", Some(gst::ClockTime::from_seconds(5)));
    {
        let input = input.get_mut().expect("writable metadata fixture");
        input.set_dts(gst::ClockTime::from_seconds(4));
        input.set_duration(gst::ClockTime::from_mseconds(250));
        input.set_offset(11);
        input.set_offset_end(23);
        input.set_flags(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER);
        let reference = gst::Caps::builder("timestamp/x-test").build();
        gst::ReferenceTimestampMeta::add(input, &reference, gst::ClockTime::from_seconds(42), None);
    }
    assert_eq!(harness.push(input), Ok(gst::FlowSuccess::Ok));
    let output = harness.pull().expect("pulling passthrough buffer");
    assert_eq!(
        output.map_readable().expect("mapping output").as_slice(),
        b"jpeg-payload"
    );
    assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(5)));
    assert_eq!(output.dts(), Some(gst::ClockTime::from_seconds(4)));
    assert_eq!(output.duration(), Some(gst::ClockTime::from_mseconds(250)));
    assert_eq!(output.offset(), 11);
    assert_eq!(output.offset_end(), 23);
    assert!(
        output
            .flags()
            .contains(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER)
    );
    assert!(output.meta::<gst::ReferenceTimestampMeta>().is_some());
}

#[test]
fn worker_protocol_one_frame_posts_exact_auth_usage_and_result() {
    let server = make_server(vec![ok_reply()]);
    let key_file = tempfile::NamedTempFile::new().expect("creating key file");
    std::fs::write(key_file.path(), "test-secret\n").expect("writing key fixture");
    let element = make_element(&server.endpoint);
    element.set_property("api-key-file", key_file.path().to_string_lossy().as_ref());
    element.set_property("system-prompt", Some("system"));
    element.set_property("user-prompt", "literal {{prompt}}");
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"one", Some(gst::ClockTime::SECOND))),
        Ok(gst::FlowSuccess::Ok)
    );
    let request = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving request");
    assert_eq!(request.method, hyper::Method::POST);
    assert_eq!(request.uri, "/v1/chat/completions");
    assert_eq!(
        request
            .headers
            .get(hyper::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-secret")
    );
    let json: serde_json::Value =
        serde_json::from_slice(&request.body).expect("parsing recorded request");
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(
        json["messages"][1]["content"][0]["text"],
        "literal {{prompt}}"
    );
    assert!(
        json["messages"][1]["content"][1]["image_url"]["url"]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/jpeg;base64,"))
    );
    let message = bus
        .timed_pop_filtered(
            gst::ClockTime::from_seconds(2),
            &[gst::MessageType::Element],
        )
        .expect("receiving result message");
    let structure = message.structure().expect("result structure");
    assert_eq!(structure.name(), "vlmanalysis-result");
    assert_eq!(structure.get::<u64>("prompt-tokens"), Ok(7));
    assert_eq!(structure.get::<u64>("completion-tokens"), Ok(3));
}

#[test]
fn worker_protocol_two_frames_without_auth_preserves_order_and_mixed_pts() {
    let server = make_server(vec![ok_reply()]);
    let endpoint = format!("{}?api-version=test", server.endpoint);
    let element = make_element(&endpoint);
    element.set_property("frames-per-request", 2_u32);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(harness.push(jpeg(b"one", None)), Ok(gst::FlowSuccess::Ok));
    assert!(matches!(
        server.requests.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(
        harness.push(jpeg(b"two", Some(gst::ClockTime::from_seconds(2)))),
        Ok(gst::FlowSuccess::Ok)
    );
    let request = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving two-frame request");
    assert_eq!(request.method, hyper::Method::POST);
    assert_eq!(request.uri, "/v1/chat/completions?api-version=test");
    assert!(request.headers.get(hyper::header::AUTHORIZATION).is_none());
    let json: serde_json::Value =
        serde_json::from_slice(&request.body).expect("parsing two-frame request");
    assert_eq!(json["messages"].as_array().map(Vec::len), Some(1));
    let content = json["messages"][0]["content"]
        .as_array()
        .expect("reading ordered multimodal content");
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(
        content[1]["image_url"]["url"],
        "data:image/jpeg;base64,b25l"
    );
    assert_eq!(
        content[2]["image_url"]["url"],
        "data:image/jpeg;base64,dHdv"
    );
    assert_eq!(json["max_tokens"], 512);
    assert_eq!(json["temperature"], 0.2);
    assert_eq!(json["top_p"], 0.9);
    assert_eq!(json["stream"], false);

    let structure = wait_for_structure(&bus, "vlmanalysis-result")
        .expect("receiving two-frame result structure");
    assert_eq!(structure.get::<u64>("request-id"), Ok(1));
    assert!(
        structure
            .get::<u64>("generation")
            .is_ok_and(|value| value >= 1)
    );
    assert_eq!(
        structure.get::<String>("model").as_deref(),
        Ok("test-model")
    );
    assert_eq!(
        structure.get::<String>("text").as_deref(),
        Ok("description")
    );
    assert_eq!(structure.get::<u32>("frame-count"), Ok(2));
    assert_eq!(
        structure.get::<gst::ClockTime>("start-pts"),
        Ok(gst::ClockTime::from_seconds(2))
    );
    assert_eq!(
        structure.get::<gst::ClockTime>("end-pts"),
        Ok(gst::ClockTime::from_seconds(2))
    );
    let latency = structure
        .get::<u64>("latency")
        .expect("result latency is an unsigned nanosecond value");
    assert!(latency <= 2_000_000_000);
}

#[test]
fn sampling_interval_boundaries_backward_jumps_no_pts_and_generation_events() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    element.set_property("analysis-interval", 5 * gst::ClockTime::SECOND.nseconds());
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_outcomes(&element, 1);
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::from_seconds(4)))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(element.property::<u64>("submitted-requests"), 1);
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::from_seconds(5)))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_outcomes(&element, 2);
    assert_eq!(
        harness.push(jpeg(b"backward", Some(gst::ClockTime::from_seconds(1)))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_outcomes(&element, 3);
    assert!(harness.push_event(gst::event::FlushStop::new(false)));
    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
    assert!(harness.push_event(gst::event::Segment::new(&segment)));
    assert_eq!(
        harness.push(jpeg(b"after-flush", Some(gst::ClockTime::from_seconds(1)))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_outcomes(&element, 4);

    let no_pts_server = make_server(vec![ok_reply()]);
    let no_pts_element = make_element(&no_pts_server.endpoint);
    no_pts_element.set_property("analysis-interval", gst::ClockTime::SECOND.nseconds());
    let mut no_pts_harness = make_harness(&no_pts_element);
    assert_eq!(
        no_pts_harness.push(jpeg(b"first", None)),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(
        no_pts_harness.push(jpeg(b"second", None)),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(no_pts_element.property::<u64>("submitted-requests"), 1);
}

#[test]
fn sampling_mixed_pts_does_not_add_no_pts_frame_to_incomplete_batch() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    element.set_property("analysis-interval", gst::ClockTime::SECOND.nseconds());
    element.set_property("frames-per-request", 2_u32);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"pts-first", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(
        harness.push(jpeg(b"no-pts-middle", None)),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(matches!(
        server.requests.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(
        harness.push(jpeg(b"pts-second", Some(gst::ClockTime::SECOND))),
        Ok(gst::FlowSuccess::Ok)
    );
    let request = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving mixed-PTS batch request");
    let request = String::from_utf8(request.body).expect("mixed-PTS request is UTF-8 JSON");
    let first = base64::engine::general_purpose::STANDARD.encode(b"pts-first");
    let middle = base64::engine::general_purpose::STANDARD.encode(b"no-pts-middle");
    let second = base64::engine::general_purpose::STANDARD.encode(b"pts-second");
    assert!(request.contains(&first));
    assert!(!request.contains(&middle));
    assert!(request.contains(&second));
    let result = wait_for_structure(&bus, "vlmanalysis-result")
        .expect("receiving mixed-PTS result structure");
    assert_eq!(result.get::<u32>("frame-count"), Ok(2));
    assert_eq!(
        result.get::<gst::ClockTime>("start-pts"),
        Ok(gst::ClockTime::ZERO)
    );
    assert_eq!(
        result.get::<gst::ClockTime>("end-pts"),
        Ok(gst::ClockTime::SECOND)
    );
}

#[test]
fn generation_stale_inflight_result_keeps_old_generation() {
    let server = make_server(vec![
        Reply {
            delay: Duration::from_millis(200),
            ..ok_reply()
        },
        ok_reply(),
    ]);
    let element = make_element(&server.endpoint);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"old", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let _old_request = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("observing old request in flight");
    assert!(harness.push_event(gst::event::FlushStop::new(false)));
    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
    assert!(harness.push_event(gst::event::Segment::new(&segment)));
    assert_eq!(
        harness.push(jpeg(b"new", Some(gst::ClockTime::SECOND))),
        Ok(gst::FlowSuccess::Ok)
    );
    let old_result =
        wait_for_structure(&bus, "vlmanalysis-result").expect("receiving stale-generation result");
    let new_result = wait_for_structure(&bus, "vlmanalysis-result")
        .expect("receiving current-generation result");
    assert_eq!(old_result.get::<u64>("request-id"), Ok(1));
    assert_eq!(new_result.get::<u64>("request-id"), Ok(2));
    let old_generation = old_result
        .get::<u64>("generation")
        .expect("old result generation");
    let new_generation = new_result
        .get::<u64>("generation")
        .expect("new result generation");
    assert!(new_generation > old_generation);
}

#[test]
fn generation_event_discards_incomplete_batch_and_eos_does_not_submit_it() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    element.set_property("frames-per-request", 2_u32);
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"discard-me", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(harness.push_event(gst::event::FlushStop::new(false)));
    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
    assert!(harness.push_event(gst::event::Segment::new(&segment)));
    assert_eq!(
        harness.push(jpeg(b"fresh-one", Some(gst::ClockTime::SECOND))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(matches!(
        server.requests.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(
        harness.push(jpeg(b"fresh-two", Some(gst::ClockTime::from_seconds(2)),)),
        Ok(gst::FlowSuccess::Ok)
    );
    let request = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving post-generation batch");
    let request_text = String::from_utf8(request.body).expect("UTF-8 JSON request");
    assert!(!request_text.contains("ZGlzY2FyZC1tZQ=="));
    assert!(request_text.contains("ZnJlc2gtb25l"));
    assert!(request_text.contains("ZnJlc2gtdHdv"));

    let eos_server = make_server(vec![ok_reply()]);
    let eos_element = make_element(&eos_server.endpoint);
    eos_element.set_property("frames-per-request", 2_u32);
    let mut eos_harness = make_harness(&eos_element);
    assert_eq!(
        eos_harness.push(jpeg(b"incomplete", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let started = Instant::now();
    assert!(eos_harness.push_event(gst::event::Eos::new()));
    assert!(started.elapsed() < Duration::from_millis(100));
    assert_eq!(eos_element.property::<u64>("submitted-requests"), 0);
}

#[test]
fn security_startup_validation_rejects_unsafe_or_invalid_settings() {
    init();
    for endpoint in [
        "http://example.com/v1/chat/completions",
        "ftp://localhost/generate",
        "http://user:password@localhost/generate",
        "http://localhost/generate#fragment",
    ] {
        let element = gst::ElementFactory::make("vlmanalysis")
            .property("endpoint", endpoint)
            .property("model", "test")
            .build()
            .expect("building validation fixture");
        assert_eq!(
            element.set_state(gst::State::Paused),
            Err(gst::StateChangeError)
        );
        let _state = element.set_state(gst::State::Null);
    }
    let missing_model = gst::ElementFactory::make("vlmanalysis")
        .build()
        .expect("building missing-model fixture");
    assert_eq!(
        missing_model.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    let _state = missing_model.set_state(gst::State::Null);

    let invalid_limits = gst::ElementFactory::make("vlmanalysis")
        .property("model", "test")
        .property("frames-per-request", 3_u32)
        .build()
        .expect("building invalid-limit fixture");
    assert_eq!(
        invalid_limits.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    let _state = invalid_limits.set_state(gst::State::Null);

    for endpoint in [
        "http://localhost:8000/v1/chat/completions",
        "http://127.0.0.2:8000/v1/chat/completions",
        "http://[::1]:8000/v1/chat/completions",
    ] {
        let accepted = gst::ElementFactory::make("vlmanalysis")
            .property("endpoint", endpoint)
            .property("model", "test")
            .build()
            .expect("building loopback fixture");
        assert!(matches!(
            accepted.set_state(gst::State::Paused),
            Ok(gst::StateChangeSuccess::Success | gst::StateChangeSuccess::NoPreroll)
        ));
        let _state = accepted.set_state(gst::State::Null);
    }
    let opted_in = gst::ElementFactory::make("vlmanalysis")
        .property("endpoint", "http://example.com/v1/chat/completions")
        .property("allow-insecure-http", true)
        .property("model", "test")
        .build()
        .expect("building explicitly insecure fixture");
    assert!(matches!(
        opted_in.set_state(gst::State::Paused),
        Ok(gst::StateChangeSuccess::Success | gst::StateChangeSuccess::NoPreroll)
    ));
    let _state = opted_in.set_state(gst::State::Null);
}

#[test]
fn security_api_key_file_failures_are_sanitized() {
    init();
    let directory = tempfile::tempdir().expect("creating secret test directory");
    let missing_path = directory.path().join("DO-NOT-LEAK-MISSING-KEY");
    let empty_path = directory.path().join("DO-NOT-LEAK-EMPTY-KEY");
    std::fs::write(&empty_path, " \n\t").expect("writing empty key fixture");
    let invalid_utf8_path = directory.path().join("DO-NOT-LEAK-INVALID-KEY");
    std::fs::write(&invalid_utf8_path, [0xff]).expect("writing invalid UTF-8 key fixture");
    for path in [missing_path, empty_path, invalid_utf8_path] {
        let element = gst::ElementFactory::make("vlmanalysis")
            .property("model", "test")
            .property("api-key-file", path.to_string_lossy().as_ref())
            .build()
            .expect("building key validation fixture");
        let bus = gst::Bus::new();
        element.set_bus(Some(&bus));
        assert_eq!(
            element.set_state(gst::State::Paused),
            Err(gst::StateChangeError)
        );
        let message = bus
            .timed_pop_filtered(gst::ClockTime::from_seconds(1), &[gst::MessageType::Error])
            .expect("receiving sanitized startup error");
        let rendered = match message.view() {
            gst::MessageView::Error(error) => {
                format!("{} {:?}", error.error(), error.debug())
            }
            _ => String::new(),
        };
        assert!(!rendered.contains("DO-NOT-LEAK"));
        let _state = element.set_state(gst::State::Null);
    }
}

#[test]
fn worker_http_errors_malformed_oversize_timeout_and_later_recovery_are_sanitized() {
    let marker = "ECHOED-BODY-MARKER";
    let api_key = "TEST-API-KEY-SECRET";
    let prompt = "TEST-PROMPT-SECRET";
    let replies = vec![
        Reply {
            status: StatusCode::BAD_REQUEST,
            body: format!("{marker}-{api_key}-{prompt}").into_bytes(),
            delay: Duration::ZERO,
            location: None,
        },
        Reply {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: marker.as_bytes().to_vec(),
            delay: Duration::ZERO,
            location: None,
        },
        Reply::json("not json"),
        Reply {
            status: StatusCode::OK,
            body: vec![b'x'; 1024 * 1024 + 1],
            delay: Duration::ZERO,
            location: None,
        },
        ok_reply(),
    ];
    let server = make_server(replies);
    let element = make_element(&server.endpoint);
    let key_file = tempfile::NamedTempFile::new().expect("creating redaction key file");
    std::fs::write(key_file.path(), api_key).expect("writing redaction key fixture");
    element.set_property("api-key-file", key_file.path().to_string_lossy().as_ref());
    element.set_property("user-prompt", prompt);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    for id in 0..5 {
        assert_eq!(
            harness.push(jpeg(b"frame", Some(gst::ClockTime::from_seconds(id)))),
            Ok(gst::FlowSuccess::Ok)
        );
        wait_for_outcomes(&element, id + 1);
        let name = if id == 4 {
            "vlmanalysis-result"
        } else {
            "vlmanalysis-error"
        };
        let structure =
            wait_for_structure(&bus, name).expect("receiving expected recovery sequence structure");
        let rendered = structure.to_string();
        for forbidden in [
            marker,
            api_key,
            prompt,
            "Authorization",
            "data:image/jpeg;base64,",
        ] {
            assert!(!rendered.contains(forbidden));
        }
        assert_eq!(structure.get::<u64>("request-id"), Ok(id + 1));
        if id < 2 {
            assert_eq!(structure.get::<String>("kind").as_deref(), Ok("http"));
            let expected_status = if id == 0 { 400 } else { 500 };
            assert_eq!(structure.get::<u32>("http-status"), Ok(expected_status));
        } else if id < 4 {
            assert_eq!(structure.get::<String>("kind").as_deref(), Ok("response"));
            assert!(!structure.has_field("http-status"));
        }
    }
    wait_for_counter(&element, "completed-requests", 1);
    assert_eq!(element.property::<u64>("failed-requests"), 4);
}

#[test]
fn worker_timeout_is_recoverable_and_push_is_nonblocking() {
    let server = make_server(vec![
        Reply {
            delay: Duration::from_millis(300),
            ..ok_reply()
        },
        ok_reply(),
    ]);
    let element = make_element(&server.endpoint);
    element.set_property("request-timeout", 50_000_000_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    let started = Instant::now();
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(started.elapsed() < Duration::from_millis(100));
    wait_for_counter(&element, "failed-requests", 1);
    let timeout =
        wait_for_structure(&bus, "vlmanalysis-error").expect("receiving timeout error structure");
    assert_eq!(timeout.get::<String>("kind").as_deref(), Ok("timeout"));
    assert!(!timeout.has_field("http-status"));
    assert_eq!(
        harness.push(jpeg(b"recovery", Some(gst::ClockTime::SECOND))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_counter(&element, "completed-requests", 1);
    let recovered = wait_for_structure(&bus, "vlmanalysis-result")
        .expect("receiving post-timeout recovery result");
    assert_eq!(recovered.get::<u64>("request-id"), Ok(2));
}

#[test]
fn backpressure_queue_drops_newest_batches_without_blocking() {
    let server = make_server(vec![Reply {
        delay: Duration::from_millis(500),
        ..ok_reply()
    }]);
    let element = make_element(&server.endpoint);
    element.set_property("queue-capacity", 1_u32);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"first", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let first = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("observing first request in flight");
    assert!(
        String::from_utf8(first.body)
            .expect("first request JSON")
            .contains("Zmlyc3Q=")
    );
    assert_eq!(
        harness.push(jpeg(b"second", Some(gst::ClockTime::from_nseconds(1)))),
        Ok(gst::FlowSuccess::Ok)
    );
    let started = Instant::now();
    assert_eq!(
        harness.push(jpeg(b"newest", Some(gst::ClockTime::from_nseconds(2)))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(started.elapsed() < Duration::from_millis(100));
    let dropped = wait_for_structure(&bus, "vlmanalysis-error")
        .expect("receiving backpressure error structure");
    assert_eq!(dropped.get::<String>("kind").as_deref(), Ok("backpressure"));
    assert_eq!(dropped.get::<u64>("request-id"), Ok(3));
    assert_eq!(dropped.get::<u32>("frame-count"), Ok(1));
    assert_eq!(element.property::<u64>("submitted-requests"), 2);
    assert_eq!(element.property::<u64>("dropped-batches"), 1);
    let second = server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving queued second request");
    let second_body = String::from_utf8(second.body).expect("second request JSON");
    assert!(second_body.contains("c2Vjb25k"));
    assert!(!second_body.contains("bmV3ZXN0"));
}

#[test]
fn redirect_policy_does_not_follow_cross_origin_location() {
    let destination = make_server(vec![ok_reply()]);
    let source = make_server(vec![Reply {
        status: StatusCode::FOUND,
        body: b"redirect".to_vec(),
        delay: Duration::ZERO,
        location: Some(destination.endpoint.clone()),
    }]);
    let element = make_element(&source.endpoint);
    let key_file = tempfile::NamedTempFile::new().expect("creating redirect key file");
    std::fs::write(key_file.path(), "redirect-secret").expect("writing redirect key fixture");
    element.set_property("api-key-file", key_file.path().to_string_lossy().as_ref());
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let source_request = source
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("receiving redirect source request");
    assert_eq!(
        source_request
            .headers
            .get(hyper::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer redirect-secret")
    );
    wait_for_counter(&element, "failed-requests", 1);
    assert!(matches!(
        destination
            .requests
            .recv_timeout(Duration::from_millis(100)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
}

#[test]
fn input_size_limits_are_recoverable_and_preserve_downstream_buffer() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    element.set_property("max-frame-bytes", 2_u64);
    element.set_property("max-batch-bytes", 2_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"oversize", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(
        harness.pull().expect("pulling oversize passthrough").size(),
        8
    );
    assert_eq!(element.property::<u64>("failed-requests"), 1);
    let error = wait_for_structure(&bus, "vlmanalysis-error")
        .expect("receiving oversize-input error structure");
    assert_eq!(error.get::<String>("kind").as_deref(), Ok("input"));
    assert!(!error.has_field("request-id"));
    assert!(!error.has_field("http-status"));
    assert_eq!(
        error.get::<String>("message").as_deref(),
        Ok("selected JPEG exceeds max-frame-bytes")
    );
    assert_eq!(error.get::<u32>("frame-count"), Ok(1));
}

#[test]
fn shutdown_immediate_and_timed_drain_are_bounded() {
    let immediate_server = make_server(vec![Reply {
        delay: Duration::from_secs(1),
        ..ok_reply()
    }]);
    let immediate_element = make_element(&immediate_server.endpoint);
    immediate_element.set_property("drain-timeout", 0_u64);
    let immediate_bus = gst::Bus::new();
    immediate_element.set_bus(Some(&immediate_bus));
    let mut immediate_harness = make_harness(&immediate_element);
    assert_eq!(
        immediate_harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let _request = immediate_server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("observing immediate-shutdown request");
    let started = Instant::now();
    drop(immediate_harness);
    assert!(started.elapsed() < Duration::from_millis(250));
    assert!(
        immediate_bus
            .timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Element])
            .is_none()
    );

    let drain_server = make_server(vec![Reply {
        delay: Duration::from_millis(40),
        ..ok_reply()
    }]);
    let drain_element = make_element(&drain_server.endpoint);
    drain_element.set_property("drain-timeout", 500_000_000_u64);
    let drain_bus = gst::Bus::new();
    drain_element.set_bus(Some(&drain_bus));
    let mut drain_harness = make_harness(&drain_element);
    assert_eq!(
        drain_harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    let _request = drain_server
        .requests
        .recv_timeout(Duration::from_secs(2))
        .expect("observing draining request");
    let started = Instant::now();
    drop(drain_harness);
    assert!(started.elapsed() < Duration::from_millis(500));
    assert_eq!(drain_element.property::<u64>("completed-requests"), 1);
    let result = wait_for_structure(&drain_bus, "vlmanalysis-result")
        .expect("receiving drained result structure");
    assert_eq!(result.get::<u64>("request-id"), Ok(1));
}

#[test]
fn property_mutability_and_counter_lifecycle_reset() {
    let server = make_server(vec![ok_reply()]);
    let element = make_element(&server.endpoint);
    let defaults = gst::ElementFactory::make("vlmanalysis")
        .build()
        .expect("building default-property fixture");
    for name in [
        "endpoint",
        "allow-insecure-http",
        "api-key-file",
        "model",
        "system-prompt",
        "user-prompt",
        "analysis-interval",
        "frames-per-request",
        "max-tokens",
        "temperature",
        "top-p",
        "request-timeout",
        "queue-capacity",
        "max-frame-bytes",
        "max-batch-bytes",
        "drain-timeout",
    ] {
        let property = element
            .find_property(name)
            .expect("finding configurable property");
        assert!(property.flags().contains(gst::PARAM_FLAG_MUTABLE_READY));
        assert!(property.flags().contains(gst::glib::ParamFlags::WRITABLE));
    }
    for name in [
        "submitted-requests",
        "completed-requests",
        "failed-requests",
        "dropped-batches",
    ] {
        let property = element
            .find_property(name)
            .expect("finding counter property");
        assert!(!property.flags().contains(gst::glib::ParamFlags::WRITABLE));
    }
    assert_eq!(element.property::<u32>("frames-per-request"), 1);
    assert_eq!(
        defaults.property::<String>("user-prompt"),
        "Describe what you see in these images."
    );
    assert_eq!(defaults.property::<u64>("analysis-interval"), 5_000_000_000);
    assert_eq!(element.property::<u64>("submitted-requests"), 0);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = make_harness(&element);
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(element.property::<u64>("submitted-requests"), 1);
    wait_for_outcomes(&element, 1);
    let first_result =
        wait_for_structure(&bus, "vlmanalysis-result").expect("receiving first lifecycle result");
    assert_eq!(first_result.get::<u64>("request-id"), Ok(1));
    assert!(matches!(
        element.set_state(gst::State::Null),
        Ok(gst::StateChangeSuccess::Success)
    ));
    harness.play();
    for counter in [
        "submitted-requests",
        "completed-requests",
        "failed-requests",
        "dropped-batches",
    ] {
        assert_eq!(element.property::<u64>(counter), 0);
    }
    assert_eq!(
        harness.push(jpeg(b"frame", Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_outcomes(&element, 1);
    let result = wait_for_structure(&bus, "vlmanalysis-result")
        .expect("receiving restarted lifecycle result");
    assert_eq!(result.get::<u64>("request-id"), Ok(1));
    assert_eq!(element.property::<u64>("submitted-requests"), 1);
    assert_eq!(element.property::<u64>("completed-requests"), 1);
}

#[test]
#[ignore = "requires VLM_TEST_ENDPOINT and VLM_TEST_MODEL"]
fn live_openai_compatible_smoke() {
    init();
    let Ok(endpoint) = std::env::var("VLM_TEST_ENDPOINT") else {
        return;
    };
    let Ok(model) = std::env::var("VLM_TEST_MODEL") else {
        return;
    };
    let mut builder = gst::ElementFactory::make("vlmanalysis")
        .property("endpoint", endpoint)
        .property("model", model);
    if let Ok(key_file) = std::env::var("VLM_TEST_API_KEY_FILE") {
        builder = builder.property("api-key-file", key_file);
    }
    let element = builder.build().expect("constructing live smoke element");
    let mut harness = make_harness(&element);
    let tiny_jpeg = base64::engine::general_purpose::STANDARD
        .decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q==")
        .expect("decoding generated 1x1 JPEG fixture");
    assert_eq!(
        harness.push(jpeg(&tiny_jpeg, Some(gst::ClockTime::ZERO))),
        Ok(gst::FlowSuccess::Ok)
    );
    wait_for_counter(&element, "completed-requests", 1);
}
