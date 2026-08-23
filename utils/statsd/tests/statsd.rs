#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests require concise assertions around bounded local resources"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Once;
use std::time::{Duration, Instant};
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
        gststatsd::plugin_register_static().expect("registering the StatsD plugin");
    });
}

#[test]
fn registers_statsd_tracer() {
    init();
    assert!(
        gst::TracerFactory::factories()
            .iter()
            .any(|factory| factory.name() == "statsd")
    );
}

#[test]
fn property_gst_tracers_environment_accepts_creation_parameters() {
    let executable = std::env::current_exe().expect("locating integration-test executable");
    let plugin_path = executable
        .parent()
        .and_then(Path::parent)
        .expect("finding Cargo target profile directory");
    let output = Command::new("gst-launch-1.0")
        .env("GST_PLUGIN_PATH", plugin_path)
        .env(
            "GST_TRACERS",
            r#"statsd(destination="127.0.0.1:9",prefix="test",global-tags="env:test",flush-interval-ms=(uint)100,exclude-filter=".*",max-pad-series=(uint)7)"#,
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

fn wait_for_eos(pipeline: &gst::Pipeline) {
    let bus = pipeline.bus().expect("pipeline bus");
    let message = bus
        .timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        )
        .expect("pipeline completion message");
    assert_eq!(message.type_(), gst::MessageType::Eos);
}

fn collect(socket: &std::net::UdpSocket, until: impl Fn(&str) -> bool) -> String {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut payload = String::new();
    while Instant::now() < deadline && !until(&payload) {
        let mut buffer = [0_u8; 1_500];
        if let Ok(size) = socket.recv(&mut buffer) {
            payload.push_str(&String::from_utf8_lossy(&buffer[..size]));
        }
    }
    payload
}

#[derive(Debug, Eq, PartialEq)]
struct ParsedMetric {
    name: String,
    value: String,
    metric_type: String,
    tags: BTreeMap<String, String>,
}

fn parse_metric(line: &str) -> Option<ParsedMetric> {
    let (metric, raw_tags) = line.split_once("|#").unwrap_or((line, ""));
    let (name, rest) = metric.split_once(':')?;
    let (value, metric_type) = rest.split_once('|')?;
    let tags = raw_tags
        .split(',')
        .filter(|tag| !tag.is_empty())
        .map(|tag| {
            tag.split_once(':')
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
        })
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(ParsedMetric {
        name: name.to_owned(),
        value: value.to_owned(),
        metric_type: metric_type.to_owned(),
        tags,
    })
}

fn parse_payload(payload: &str) -> Vec<ParsedMetric> {
    payload.lines().filter_map(parse_metric).collect()
}

fn counter_sum(payload: &str, key: &str) -> i64 {
    parse_payload(payload)
        .iter()
        .filter(|metric| metric.name == key && metric.metric_type == "c")
        .filter_map(|metric| metric.value.parse::<i64>().ok())
        .sum()
}

fn loopback() -> (std::net::UdpSocket, String) {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("binding UDP receiver");
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .expect("setting receiver timeout");
    let address = socket.local_addr().expect("receiver address").to_string();
    (socket, address)
}

fn assert_reference_metrics(payload: &str) {
    assert_eq!(
        counter_sum(payload, "gstsmith.gstreamer.pad.push_buffers"),
        3,
        "{payload}"
    );
    let metrics = parse_payload(payload);
    let pad_buffers = metrics
        .iter()
        .filter(|metric| {
            metric.name == "gstsmith.gstreamer.pad.push_buffers"
                && metric.metric_type == "c"
                && metric.tags.len() == 2
                && metric
                    .tags
                    .get("element")
                    .is_some_and(|tag| tag.contains("observed"))
                && metric.tags.get("pad").is_some_and(|tag| tag == "src")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pad_buffers
            .iter()
            .filter_map(|metric| metric.value.parse::<i64>().ok())
            .sum::<i64>(),
        3,
        "{metrics:?}"
    );
    let pad_bytes = metrics
        .iter()
        .filter(|metric| {
            metric.name == "gstsmith.gstreamer.pad.push_bytes"
                && metric.metric_type == "c"
                && metric.tags.len() == 2
                && metric
                    .tags
                    .get("element")
                    .is_some_and(|tag| tag.contains("observed"))
                && metric.tags.get("pad").is_some_and(|tag| tag == "src")
        })
        .filter_map(|metric| metric.value.parse::<i64>().ok())
        .sum::<i64>();
    assert!(pad_bytes > 0, "{metrics:?}");

    for name in [
        "gstsmith.gstreamer.queue.level_buffers",
        "gstsmith.gstreamer.queue.level_bytes",
        "gstsmith.gstreamer.queue.level_seconds",
        "gstsmith.gstreamer.queue.capacity_buffers",
        "gstsmith.gstreamer.queue.capacity_bytes",
        "gstsmith.gstreamer.queue.capacity_seconds",
    ] {
        assert!(
            metrics.iter().any(|metric| {
                metric.name == name
                    && metric.metric_type == "g"
                    && metric.value.parse::<f64>().is_ok()
                    && metric.tags.len() == 1
                    && metric
                        .tags
                        .get("element")
                        .is_some_and(|tag| tag.contains("observed"))
            }),
            "missing exact queue gauge {name}: {metrics:?}"
        );
    }

    let mut states = BTreeMap::new();
    for metric in metrics.iter().filter(|metric| {
        metric.name == "gstsmith.gstreamer.pipeline.state"
            && metric.metric_type == "g"
            && metric.tags.len() == 2
            && metric
                .tags
                .get("pipeline")
                .is_some_and(|pipeline| pipeline == "reference")
    }) {
        if let (Some(state), Ok(value)) = (metric.tags.get("state"), metric.value.parse::<u64>()) {
            states.insert(state.clone(), value);
        }
    }
    assert_eq!(
        states.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        BTreeSet::from(["null", "paused", "playing", "ready"]),
        "{metrics:?}"
    );
    assert!(states.values().all(|value| *value <= 1), "{metrics:?}");
    assert_eq!(states.values().sum::<u64>(), 1, "{metrics:?}");
}

#[test]
fn reference_pipeline_exports_counters_and_gauges() {
    init();
    let (socket, destination) = loopback();
    let tracer = gst::glib::Object::builder::<gststatsd::statsd::StatsdTracer>()
        .property("destination", &destination)
        .property("flush-interval-ms", 100_u32)
        .property(
            "include-filter",
            "^reference$|GstQueue:observed$|observed:src$",
        )
        .build();
    assert!(tracer.property::<bool>("worker-running"));
    let pipeline = gst::parse::launch(
        "videotestsrc num-buffers=3 ! queue name=observed ! fakesink sync=false",
    )
    .expect("constructing reference pipeline")
    .downcast::<gst::Pipeline>()
    .expect("launch returned pipeline");
    pipeline.set_property("name", "reference");
    pipeline
        .set_state(gst::State::Playing)
        .expect("starting pipeline");
    wait_for_eos(&pipeline);
    let payload = collect(&socket, |payload| {
        payload.contains("gstreamer.queue.capacity_buffers")
            && payload.contains("gstreamer.pipeline.state")
            && counter_sum(payload, "gstsmith.gstreamer.pad.push_buffers") >= 3
    });
    pipeline
        .set_state(gst::State::Null)
        .expect("stopping pipeline");
    assert_reference_metrics(&payload);
    drop(tracer);
}

#[test]
fn series_limit_pipeline_is_bounded() {
    init();
    let (socket, destination) = loopback();
    let tracer = gst::glib::Object::builder::<gststatsd::statsd::StatsdTracer>()
        .property("destination", &destination)
        .property("flush-interval-ms", 100_u32)
        .property("max-pad-series", 1_u32)
        .property("include-filter", "first:src$|second:src$")
        .build();
    let pipeline = gst::parse::launch(
        "fakesrc num-buffers=3 ! identity name=first ! identity name=second ! fakesink sync=false",
    )
    .expect("constructing capped pipeline")
    .downcast::<gst::Pipeline>()
    .expect("launch returned pipeline");
    pipeline
        .set_state(gst::State::Playing)
        .expect("starting pipeline");
    wait_for_eos(&pipeline);
    let payload = collect(&socket, |payload| {
        counter_sum(payload, "gstsmith.gstreamer.untracked_pad_events") >= 3
    });
    pipeline
        .set_state(gst::State::Null)
        .expect("stopping pipeline");
    let metrics = parse_payload(&payload);
    let pad_lines = metrics
        .iter()
        .filter(|metric| {
            metric.name == "gstsmith.gstreamer.pad.push_buffers"
                && metric.metric_type == "c"
                && metric.tags.len() == 2
                && metric.tags.get("pad").is_some_and(|tag| tag == "src")
        })
        .collect::<Vec<_>>();
    assert!(!pad_lines.is_empty(), "{payload}");
    let has_first = pad_lines.iter().any(|metric| {
        metric
            .tags
            .get("element")
            .is_some_and(|tag| tag.contains("first"))
    });
    let has_second = pad_lines.iter().any(|metric| {
        metric
            .tags
            .get("element")
            .is_some_and(|tag| tag.contains("second"))
    });
    assert_ne!(
        has_first, has_second,
        "only one pad labelset is allowed: {payload}"
    );
    let diagnostic = metrics
        .iter()
        .filter(|metric| {
            metric.name == "gstsmith.gstreamer.untracked_pad_events"
                && metric.metric_type == "c"
                && metric.tags == BTreeMap::from([("reason".to_owned(), "series_limit".to_owned())])
        })
        .filter_map(|metric| metric.value.parse::<i64>().ok())
        .sum::<i64>();
    assert!(diagnostic >= 3, "{metrics:?}");
    drop(tracer);
}

#[test]
fn unavailable_receiver_does_not_block_shutdown() {
    init();
    let tracer = gst::glib::Object::builder::<gststatsd::statsd::StatsdTracer>()
        .property("destination", "127.0.0.1:9")
        .property("flush-interval-ms", 100_u32)
        .build();
    let pipeline = gst::parse::launch("fakesrc num-buffers=1 ! fakesink sync=false")
        .expect("constructing pipeline")
        .downcast::<gst::Pipeline>()
        .expect("launch returned pipeline");
    pipeline
        .set_state(gst::State::Playing)
        .expect("starting pipeline");
    wait_for_eos(&pipeline);
    pipeline
        .set_state(gst::State::Null)
        .expect("stopping pipeline");
    let started = Instant::now();
    drop(tracer);
    assert!(started.elapsed() < Duration::from_secs(2));
}
