#![expect(
    clippy::expect_used,
    reason = "real transport test setup and assertions require successful external operations"
)]

mod common;

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use gst::prelude::*;
use s2_sdk::types::{
    AppendInput, AppendRecord, AppendRecordBatch, BasinName, CommandRecord, EnsureBasinInput,
    EnsureStreamInput, FencingToken, ReadInput, ReadLimits, ReadStop, StreamName,
};
use s2_testcontainers::{DEFAULT_ACCESS_TOKEN, S2Lite};

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("creating S2 Lite test runtime")
}

fn unique_names(label: &str) -> (BasinName, StreamName) {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    (
        format!("gstsmith-{label}-{}-{id}", std::process::id())
            .parse()
            .expect("valid unique basin name"),
        format!("{label}-{id}")
            .parse()
            .expect("valid unique stream name"),
    )
}

fn token_file(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "gstsmith-s2-{label}-{}-{}.token",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    std::fs::write(&path, format!("{DEFAULT_ACCESS_TOKEN}\n")).expect("writing test token file");
    path
}

fn ensure_stream(
    runtime: &tokio::runtime::Runtime,
    client: &s2_sdk::S2,
    basin_name: &BasinName,
    stream_name: &StreamName,
) {
    runtime
        .block_on(client.ensure_basin(EnsureBasinInput::new(basin_name.clone())))
        .expect("ensuring test basin");
    runtime
        .block_on(
            client
                .basin(basin_name.clone())
                .ensure_stream(EnsureStreamInput::new(stream_name.clone())),
        )
        .expect("ensuring test stream");
}

fn configure(
    element: &gst::Element,
    lite: &S2Lite,
    basin: &BasinName,
    stream: &StreamName,
    token: &std::path::Path,
) {
    element.set_property("basin", basin.as_ref());
    element.set_property("stream", stream.as_ref());
    element.set_property("access-token-file", token.to_string_lossy().as_ref());
    element.set_property("account-endpoint", lite.endpoint());
    element.set_property("basin-endpoint", lite.endpoint());
}

fn header(name: Vec<u8>, value: Vec<u8>) -> gst::glib::SendValue {
    gst::Structure::builder("s2-header")
        .field("name", gst::glib::Bytes::from_owned(name))
        .field("value", gst::glib::Bytes::from_owned(value))
        .build()
        .to_send_value()
}

fn meta_buffer(body: Vec<u8>, timestamp: u64) -> gst::Buffer {
    let mut buffer = gst::Buffer::from_mut_slice(body);
    let mut meta = gst::meta::CustomMeta::add(
        buffer.get_mut().expect("new buffer is writable"),
        "GstS2RecordMeta",
    )
    .expect("adding S2 record metadata");
    let structure = meta.mut_structure();
    structure.set("basin", "ignored-source-basin");
    structure.set("stream", "ignored-source-stream");
    structure.set("seq-num", 987_u64);
    structure.set("timestamp", timestamp);
    structure.set("is-command", false);
    structure.set(
        "headers",
        gst::Array::new([
            header(vec![0, 255], vec![1, 0]),
            header(vec![0, 255], Vec::new()),
        ]),
    );
    buffer
}

fn body(buffer: &gst::Buffer) -> Vec<u8> {
    buffer
        .map_readable()
        .expect("mapping S2 buffer")
        .as_slice()
        .to_vec()
}

fn headers(buffer: &gst::Buffer) -> Vec<(Vec<u8>, Vec<u8>)> {
    let meta =
        gst::meta::CustomMeta::from_buffer(buffer, "GstS2RecordMeta").expect("S2 record metadata");
    meta.structure()
        .get::<gst::Array>("headers")
        .expect("S2 record headers")
        .iter()
        .map(|value| {
            let structure = value.get::<gst::Structure>().expect("S2 header structure");
            assert_eq!(structure.name(), "s2-header");
            let name = structure
                .get::<gst::glib::Bytes>("name")
                .expect("S2 header name");
            let value = structure
                .get::<gst::glib::Bytes>("value")
                .expect("S2 header value");
            (name.as_ref().to_vec(), value.as_ref().to_vec())
        })
        .collect()
}

fn sink_harness(element: &gst::Element, stream_id: &str) -> gst_check::Harness {
    let mut harness = gst_check::Harness::with_element(element, Some("sink"), None);
    harness.play();
    assert!(harness.push_event(gst::event::StreamStart::new(stream_id)));
    let caps = gst::Caps::builder("application/octet-stream").build();
    assert!(harness.push_event(gst::event::Caps::new(&caps)));
    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
    assert!(harness.push_event(gst::event::Segment::new(&segment)));
    harness
}

fn stop_lite(runtime: &tokio::runtime::Runtime, lite: S2Lite) {
    runtime.block_on(async move {
        drop(lite);
        tokio::task::yield_now().await;
    });
}

fn pull_from(
    lite: &S2Lite,
    basin: &BasinName,
    stream: &StreamName,
    token: &std::path::Path,
    mode: &str,
    property: Option<(&str, u64)>,
) -> gst::Buffer {
    let source = common::element("s2src");
    configure(&source, lite, basin, stream, token);
    source.set_property_from_str("start-mode", mode);
    if let Some((name, value)) = property {
        source.set_property(name, value);
    }
    let mut harness = gst_check::Harness::with_element(&source, None, Some("src"));
    harness.play();
    harness.pull().expect("pulling positioned S2 record")
}

#[test]
#[ignore = "requires Docker and the pinned S2 Lite image"]
#[expect(
    clippy::too_many_lines,
    reason = "one serial end-to-end scenario keeps the container, resources, and durability boundary together"
)]
fn sink_source_round_trip_durability_metadata_and_command_rejection() {
    common::init();
    let runtime = runtime();
    let lite = runtime.block_on(S2Lite::start()).expect("starting S2 Lite");
    let client = lite.client().expect("S2 Lite client");
    let (basin_name, stream_name) = unique_names("roundtrip");
    runtime
        .block_on(client.ensure_basin(EnsureBasinInput::new(basin_name.clone())))
        .expect("ensuring test basin");
    let basin = client.basin(basin_name.clone());
    runtime
        .block_on(basin.ensure_stream(EnsureStreamInput::new(stream_name.clone())))
        .expect("ensuring test stream");
    let token = token_file("roundtrip");

    let sink = common::element("s2sink");
    configure(&sink, &lite, &basin_name, &stream_name, &token);
    sink.set_property("preserve-timestamp", true);
    let mut input_harness = sink_harness(&sink, "roundtrip-input");
    input_harness
        .push(meta_buffer(vec![0, 255, 17, 0], 123))
        .expect("pushing metadata-bearing binary record");
    input_harness
        .push(gst::Buffer::from_mut_slice(Vec::<u8>::new()))
        .expect("pushing empty record");
    assert!(
        input_harness.push_event(gst::event::Eos::new()),
        "EOS waits for durable acknowledgements"
    );

    let records = runtime
        .block_on(
            basin.stream(stream_name.clone()).read(
                ReadInput::new()
                    .with_stop(ReadStop::new().with_limits(ReadLimits::new().with_count(2))),
            ),
        )
        .expect("reading sink output");
    assert_eq!(records.records.len(), 2);
    let first_record = records.records.first().expect("first appended record");
    let second_record = records.records.get(1).expect("second appended record");
    assert_eq!(first_record.body.as_ref(), &[0, 255, 17, 0]);
    assert_eq!(first_record.timestamp, 123);
    assert_eq!(first_record.headers.len(), 2);
    assert_eq!(
        first_record
            .headers
            .first()
            .expect("first binary header")
            .name
            .as_ref(),
        &[0, 255]
    );
    assert_eq!(
        first_record
            .headers
            .get(1)
            .expect("second binary header")
            .name
            .as_ref(),
        &[0, 255]
    );
    assert!(second_record.body.is_empty());

    assert_eq!(
        body(&pull_from(
            &lite,
            &basin_name,
            &stream_name,
            &token,
            "earliest",
            None,
        )),
        vec![0, 255, 17, 0]
    );
    assert!(
        body(&pull_from(
            &lite,
            &basin_name,
            &stream_name,
            &token,
            "sequence",
            Some(("start-seq-num", 1)),
        ))
        .is_empty()
    );
    assert_eq!(
        body(&pull_from(
            &lite,
            &basin_name,
            &stream_name,
            &token,
            "timestamp",
            Some(("start-timestamp", 123)),
        )),
        vec![0, 255, 17, 0]
    );
    assert!(
        body(&pull_from(
            &lite,
            &basin_name,
            &stream_name,
            &token,
            "tail-offset",
            Some(("tail-offset", 1)),
        ))
        .is_empty()
    );

    let command = AppendRecord::from(CommandRecord::fence(
        FencingToken::from_str("integration-fence").expect("valid fencing token"),
    ));
    let command_batch =
        AppendRecordBatch::try_from_iter([command]).expect("valid command record batch");
    runtime
        .block_on(
            basin
                .stream(stream_name.clone())
                .append(AppendInput::new(command_batch)),
        )
        .expect("appending command record");

    let source = common::element("s2src");
    configure(&source, &lite, &basin_name, &stream_name, &token);
    source.set_property_from_str("start-mode", "sequence");
    source.set_property("start-seq-num", 0_u64);
    let mut source_harness = gst_check::Harness::with_element(&source, None, Some("src"));
    source_harness.play();
    let first = source_harness.pull().expect("pulling binary source record");
    let second = source_harness.pull().expect("pulling empty source record");
    let command_buffer = source_harness
        .pull()
        .expect("pulling command source record");
    assert_eq!(body(&first), vec![0, 255, 17, 0]);
    assert!(body(&second).is_empty());
    let first_meta = gst::meta::CustomMeta::from_buffer(&first, "GstS2RecordMeta")
        .expect("source record metadata");
    assert_eq!(
        first_meta
            .structure()
            .get::<u64>("seq-num")
            .expect("source sequence"),
        0
    );
    assert_eq!(
        first_meta
            .structure()
            .get::<String>("basin")
            .expect("source basin"),
        basin_name.as_ref()
    );
    assert_eq!(
        first_meta
            .structure()
            .get::<String>("stream")
            .expect("source stream"),
        stream_name.as_ref()
    );
    assert_eq!(
        headers(&first),
        vec![(vec![0, 255], vec![1, 0]), (vec![0, 255], Vec::new())]
    );
    let command_meta = gst::meta::CustomMeta::from_buffer(&command_buffer, "GstS2RecordMeta")
        .expect("command metadata");
    assert!(
        command_meta
            .structure()
            .get::<bool>("is-command")
            .expect("command marker")
    );
    assert_eq!(body(&command_buffer), b"integration-fence");
    assert_eq!(
        headers(&command_buffer),
        vec![(Vec::new(), b"fence".to_vec())]
    );

    let (destination_basin, destination_stream) = unique_names("destination");
    runtime
        .block_on(client.ensure_basin(EnsureBasinInput::new(destination_basin.clone())))
        .expect("ensuring destination basin");
    runtime
        .block_on(
            client
                .basin(destination_basin.clone())
                .ensure_stream(EnsureStreamInput::new(destination_stream.clone())),
        )
        .expect("ensuring destination stream");
    let relay_sink = common::element("s2sink");
    configure(
        &relay_sink,
        &lite,
        &destination_basin,
        &destination_stream,
        &token,
    );
    let mut relay_harness = sink_harness(&relay_sink, "relay-input");
    relay_harness
        .push(first.copy())
        .expect("relaying source buffer to property-selected destination");
    assert!(
        relay_harness.push_event(gst::event::Eos::new()),
        "relay EOS waits for durable acknowledgement"
    );
    let relayed = runtime
        .block_on(
            client
                .basin(destination_basin.clone())
                .stream(destination_stream.clone())
                .read(
                    ReadInput::new()
                        .with_stop(ReadStop::new().with_limits(ReadLimits::new().with_count(1))),
                ),
        )
        .expect("reading relayed destination record");
    let relayed = relayed.records.first().expect("relayed destination record");
    assert_eq!(relayed.body.as_ref(), &[0, 255, 17, 0]);
    assert_eq!(relayed.headers.len(), 2);
    assert_eq!(relayed.headers[0].name.as_ref(), &[0, 255]);
    assert_eq!(relayed.headers[0].value.as_ref(), &[1, 0]);
    assert_eq!(relayed.headers[1].name.as_ref(), &[0, 255]);
    assert!(relayed.headers[1].value.is_empty());

    let command_sink = common::element("s2sink");
    configure(
        &command_sink,
        &lite,
        &destination_basin,
        &destination_stream,
        &token,
    );
    let mut command_harness = sink_harness(&command_sink, "command-rejection-input");
    assert!(
        command_harness.push(command_buffer).is_err(),
        "s2sink rejects command metadata before append"
    );
    let mut inconsistent = meta_buffer(vec![3], 0);
    gst::meta::CustomMeta::from_mut_buffer(
        inconsistent
            .get_mut()
            .expect("new inconsistent buffer is writable"),
        "GstS2RecordMeta",
    )
    .expect("finding inconsistent metadata")
    .mut_structure()
    .set("is-command", true);
    assert!(
        command_harness.push(inconsistent).is_err(),
        "s2sink rejects inconsistent command metadata before append"
    );

    drop(command_harness);
    drop(relay_harness);
    drop(source_harness);
    drop(input_harness);
    std::fs::remove_file(token).expect("removing test token file");
    stop_lite(&runtime, lite);
}

#[test]
#[ignore = "requires Docker and the pinned S2 Lite image"]
fn normal_stop_drains_accepted_records() {
    common::init();
    let runtime = runtime();
    let lite = runtime.block_on(S2Lite::start()).expect("starting S2 Lite");
    let client = lite.client().expect("S2 Lite client");
    let (basin_name, stream_name) = unique_names("normal-stop");
    ensure_stream(&runtime, &client, &basin_name, &stream_name);
    let token = token_file("normal-stop");

    let sink = common::element("s2sink");
    configure(&sink, &lite, &basin_name, &stream_name, &token);
    let mut harness = sink_harness(&sink, "normal-stop-input");
    harness
        .push(gst::Buffer::from_mut_slice(vec![0, 1, 0, 255]))
        .expect("accepting record before normal stop");
    sink.set_state(gst::State::Null)
        .expect("normal sink shutdown");

    let records = runtime
        .block_on(
            client
                .basin(basin_name.clone())
                .stream(stream_name.clone())
                .read(
                    ReadInput::new()
                        .with_stop(ReadStop::new().with_limits(ReadLimits::new().with_count(1))),
                ),
        )
        .expect("reading record drained during normal stop");
    assert_eq!(records.records.len(), 1);
    assert_eq!(records.records[0].body.as_ref(), &[0, 1, 0, 255]);

    drop(harness);
    std::fs::remove_file(token).expect("removing access-token file");
    stop_lite(&runtime, lite);
}

#[test]
#[ignore = "requires Docker and the pinned S2 Lite image"]
fn idle_tail_source_cancellation_is_bounded() {
    common::init();
    let runtime = runtime();
    let lite = runtime.block_on(S2Lite::start()).expect("starting S2 Lite");
    let client = lite.client().expect("S2 Lite client");
    let (basin_name, stream_name) = unique_names("idle");
    runtime
        .block_on(client.ensure_basin(EnsureBasinInput::new(basin_name.clone())))
        .expect("ensuring idle basin");
    runtime
        .block_on(
            client
                .basin(basin_name.clone())
                .ensure_stream(EnsureStreamInput::new(stream_name.clone())),
        )
        .expect("ensuring idle stream");
    let token = token_file("idle");
    let unwritten = common::element("s2src");
    configure(&unwritten, &lite, &basin_name, &stream_name, &token);
    unwritten.set_property_from_str("start-mode", "sequence");
    unwritten.set_property("start-seq-num", 999_u64);
    let unwritten_pipeline = gst::Pipeline::new();
    let unwritten_sink = gst::ElementFactory::make("fakesink")
        .build()
        .expect("constructing unwritten-position sink");
    unwritten_pipeline
        .add_many([&unwritten, &unwritten_sink])
        .expect("adding unwritten-position elements");
    unwritten
        .link(&unwritten_sink)
        .expect("linking unwritten-position pipeline");
    unwritten_pipeline
        .set_state(gst::State::Playing)
        .expect("starting unwritten-position pipeline");
    let error = unwritten_pipeline
        .bus()
        .expect("unwritten-position pipeline bus")
        .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error]);
    unwritten_pipeline
        .set_state(gst::State::Null)
        .expect("stopping unwritten-position source");
    assert!(
        error.is_some(),
        "unwritten source position errors without clamping"
    );

    let source = common::element("s2src");
    configure(&source, &lite, &basin_name, &stream_name, &token);
    source.set_property_from_str("start-mode", "sequence");
    source.set_property("start-seq-num", 999_u64);
    source.set_property("clamp-to-tail", true);
    let mut harness = gst_check::Harness::with_element(&source, None, Some("src"));
    harness.play();
    std::thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    source
        .set_state(gst::State::Null)
        .expect("cancelling idle source");
    assert!(started.elapsed() < Duration::from_secs(2));
    drop(harness);
    std::fs::remove_file(token).expect("removing test token file");
    stop_lite(&runtime, lite);
}

#[test]
#[ignore = "requires Docker and the pinned S2 Lite image"]
fn append_precondition_failures_are_terminal() {
    common::init();
    let runtime = runtime();
    let lite = runtime.block_on(S2Lite::start()).expect("starting S2 Lite");
    let client = lite.client().expect("S2 Lite client");
    let token = token_file("preconditions");

    let (match_basin, match_stream) = unique_names("match-failure");
    ensure_stream(&runtime, &client, &match_basin, &match_stream);
    let match_sink = common::element("s2sink");
    configure(&match_sink, &lite, &match_basin, &match_stream, &token);
    match_sink.set_property("match-seq-num-enabled", true);
    match_sink.set_property("match-seq-num", 99_u64);
    let mut match_harness = sink_harness(&match_sink, "match-failure-input");
    match_harness
        .push(gst::Buffer::from_mut_slice(vec![1]))
        .expect("locally queueing match-sequence test record");
    assert!(
        !match_harness.push_event(gst::event::Eos::new()),
        "wrong match sequence must fail the EOS durability barrier"
    );
    drop(match_harness);

    let (fence_basin, fence_stream) = unique_names("fence-failure");
    ensure_stream(&runtime, &client, &fence_basin, &fence_stream);
    let fencing_path =
        std::env::temp_dir().join(format!("gstsmith-s2-fence-{}.token", std::process::id()));
    std::fs::write(&fencing_path, "wrong-fence\n").expect("writing fencing-token file");
    let fence_sink = common::element("s2sink");
    configure(&fence_sink, &lite, &fence_basin, &fence_stream, &token);
    fence_sink.set_property(
        "fencing-token-file",
        fencing_path.to_string_lossy().as_ref(),
    );
    let mut fence_harness = sink_harness(&fence_sink, "fence-failure-input");
    fence_harness
        .push(gst::Buffer::from_mut_slice(vec![2]))
        .expect("locally queueing fencing test record");
    assert!(
        !fence_harness.push_event(gst::event::Eos::new()),
        "wrong fencing token must fail the EOS durability barrier"
    );
    drop(fence_harness);

    std::fs::remove_file(fencing_path).expect("removing fencing-token file");
    std::fs::remove_file(token).expect("removing access-token file");
    stop_lite(&runtime, lite);
}
