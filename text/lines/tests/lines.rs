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
        gstlines::plugin_register_static().expect("registering the lines plugin");
    });
}

fn element(factory: &str) -> gst::Element {
    init();
    gst::ElementFactory::make(factory)
        .build()
        .expect("constructing lines element")
}

fn parser_with_properties(
    properties: &[(&str, &dyn gst::glib::value::ToValue)],
) -> gst_check::Harness {
    init();
    let mut builder = gst::ElementFactory::make("lineparse");
    for (name, value) in properties {
        builder = builder.property(name, value.to_value());
    }
    let parser = builder.build().expect("constructing lineparse");
    let mut harness = gst_check::Harness::with_element(&parser, Some("sink"), Some("src"));
    harness.set_src_caps_str("application/octet-stream");
    harness.play();
    harness
}

fn parser() -> gst_check::Harness {
    parser_with_properties(&[])
}

fn encoder_with_delimiter(delimiter: Option<&str>) -> gst_check::Harness {
    init();
    let mut builder = gst::ElementFactory::make("lineenc");
    if let Some(delimiter) = delimiter {
        builder = builder.property("delimiter", delimiter);
    }
    let encoder = builder.build().expect("constructing lineenc");
    let mut harness = gst_check::Harness::with_element(&encoder, Some("sink"), Some("src"));
    harness.set_src_caps_str("application/octet-stream");
    harness.play();
    harness
}

fn buffer(bytes: &[u8]) -> gst::Buffer {
    gst::Buffer::from_mut_slice(bytes.to_vec())
}

fn bytes(buffer: &gst::Buffer) -> Vec<u8> {
    buffer
        .map_readable()
        .expect("mapping output buffer")
        .as_slice()
        .to_vec()
}

fn pull_bytes(harness: &mut gst_check::Harness) -> Vec<u8> {
    bytes(&harness.pull().expect("pulling output buffer"))
}

#[test]
fn lineparse_registers_plugin_and_both_factories() {
    init();

    assert!(gst::Registry::get().find_plugin("lines").is_some());
    assert!(gst::ElementFactory::find("lineparse").is_some());
    assert!(gst::ElementFactory::find("lineenc").is_some());
}

#[test]
fn lineparse_and_lineenc_advertise_any_caps() {
    for factory_name in ["lineparse", "lineenc"] {
        let element = element(factory_name);
        for pad_name in ["sink", "src"] {
            let pad = element.static_pad(pad_name).expect("finding element pad");
            assert!(
                pad.pad_template_caps().is_any(),
                "{factory_name}:{pad_name} must advertise ANY caps"
            );
        }
    }

    for factory_name in ["lineparse", "lineenc"] {
        let element = element(factory_name);
        let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
        let caps = gst::Caps::builder("application/x-lines-test")
            .field("variant", "fixed")
            .build();
        harness.set_src_caps(caps.clone());
        harness.play();
        assert_eq!(
            element
                .static_pad("src")
                .expect("finding element source pad")
                .current_caps(),
            Some(caps),
            "{factory_name} must preserve fixed upstream caps"
        );
    }
}

#[test]
fn lineparse_properties_default_and_round_trip_in_ready() {
    let parser = element("lineparse");
    assert_eq!(parser.property::<String>("delimiter"), "\n");
    assert_eq!(parser.property::<u32>("max-record-size"), 65_536);
    assert!(!parser.property::<bool>("omit-empty"));

    assert_ne!(
        parser.set_state(gst::State::Ready),
        Err(gst::StateChangeError)
    );
    parser.set_property("delimiter", "::");
    parser.set_property("max-record-size", 123_u32);
    parser.set_property("omit-empty", true);
    assert_eq!(parser.property::<String>("delimiter"), "::");
    assert_eq!(parser.property::<u32>("max-record-size"), 123);
    assert!(parser.property::<bool>("omit-empty"));
    parser
        .set_state(gst::State::Null)
        .expect("returning parser to NULL");
}

#[test]
fn lineparse_rejects_empty_delimiter_during_state_setup() {
    init();
    let parser = gst::ElementFactory::make("lineparse")
        .property("delimiter", "")
        .build()
        .expect("constructing lineparse");
    assert_eq!(
        parser.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    parser
        .set_state(gst::State::Null)
        .expect("returning parser to NULL after rejected state change");
}

#[test]
fn lineparse_frames_several_records_from_one_buffer() {
    let mut harness = parser();
    assert_eq!(
        harness.push(buffer(b"one\ntwo\nthree\n")),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(pull_bytes(&mut harness), b"one");
    assert_eq!(pull_bytes(&mut harness), b"two");
    assert_eq!(pull_bytes(&mut harness), b"three");
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_frames_a_record_split_across_buffers_and_preserves_timing() {
    let mut harness = parser();
    let mut first = buffer(b"sp");
    {
        let first = first.get_mut().expect("writable input buffer");
        first.set_pts(gst::ClockTime::from_seconds(3));
        first.set_flags(gst::BufferFlags::DISCONT);
    }

    assert_eq!(harness.push(first), Ok(gst::FlowSuccess::Ok));
    assert_eq!(harness.buffers_in_queue(), 0);
    assert_eq!(harness.push(buffer(b"lit\n")), Ok(gst::FlowSuccess::Ok));
    let output = harness.pull().expect("pulling split record");
    assert_eq!(bytes(&output), b"split");
    assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(3)));
    assert!(output.flags().contains(gst::BufferFlags::DISCONT));
}

#[test]
fn lineparse_finds_multibyte_delimiter_across_buffer_boundaries() {
    let delimiter = "::";
    let mut harness = parser_with_properties(&[("delimiter", &delimiter)]);
    assert_eq!(harness.push(buffer(b"alpha:")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(harness.buffers_in_queue(), 0);
    assert_eq!(harness.push(buffer(b":beta::")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"alpha");
    assert_eq!(pull_bytes(&mut harness), b"beta");
}

#[test]
fn lineparse_finds_multibyte_delimiter_at_every_split_position() {
    let delimiter = "<END>";
    let framed = b"alpha<END>";

    for split in 1..framed.len() {
        let mut harness = parser_with_properties(&[("delimiter", &delimiter)]);
        assert_eq!(
            harness.push(buffer(&framed[..split])),
            Ok(gst::FlowSuccess::Ok),
            "first fragment failed at split {split}"
        );
        assert_eq!(
            harness.push(buffer(&framed[split..])),
            Ok(gst::FlowSuccess::Ok),
            "second fragment failed at split {split}"
        );
        assert_eq!(
            pull_bytes(&mut harness),
            b"alpha",
            "wrong record at split {split}"
        );
        assert_eq!(harness.buffers_in_queue(), 0);
    }
}

#[test]
fn lineparse_frames_long_record_fragmented_one_byte_at_a_time() {
    let delimiter = "::";
    let mut harness = parser_with_properties(&[("delimiter", &delimiter)]);
    let payload = vec![b'a'; 4_096];

    for &byte in &payload {
        assert_eq!(harness.push(buffer(&[byte])), Ok(gst::FlowSuccess::Ok));
    }
    for &byte in delimiter.as_bytes() {
        assert_eq!(harness.push(buffer(&[byte])), Ok(gst::FlowSuccess::Ok));
    }

    assert_eq!(pull_bytes(&mut harness), payload);
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_emits_adjacent_empty_records_by_default() {
    let mut harness = parser();
    assert_eq!(harness.push(buffer(b"a\n\nb\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"a");
    assert_eq!(pull_bytes(&mut harness), b"");
    assert_eq!(pull_bytes(&mut harness), b"b");
}

#[test]
fn lineparse_omit_empty_drops_adjacent_empty_records() {
    let omit_empty = true;
    let mut harness = parser_with_properties(&[("omit-empty", &omit_empty)]);
    assert_eq!(harness.push(buffer(b"a\n\nb\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"a");
    assert_eq!(pull_bytes(&mut harness), b"b");
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_eos_emits_final_unterminated_record() {
    let mut harness = parser();
    assert_eq!(
        harness.push(buffer(b"unterminated")),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(harness.buffers_in_queue(), 0);
    assert!(harness.push_event(gst::event::Eos::new()));
    assert_eq!(pull_bytes(&mut harness), b"unterminated");
}

#[test]
fn lineparse_trailing_delimiter_has_no_phantom_record() {
    let mut harness = parser();
    assert_eq!(harness.push(buffer(b"only\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"only");
    assert!(harness.push_event(gst::event::Eos::new()));
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_empty_stream_emits_nothing() {
    let mut harness = parser();
    assert!(harness.push_event(gst::event::Eos::new()));
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_accepts_record_exactly_at_maximum() {
    let maximum = 3_u32;
    let mut harness = parser_with_properties(&[("max-record-size", &maximum)]);
    assert_eq!(harness.push(buffer(b"abc\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"abc");
}

#[test]
fn lineparse_rejects_record_one_byte_over_maximum() {
    let maximum = 3_u32;
    let mut harness = parser_with_properties(&[("max-record-size", &maximum)]);
    assert_eq!(harness.push(buffer(b"abcd")), Err(gst::FlowError::Error));
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineparse_accepts_maximum_record_with_split_delimiter() {
    let maximum = 3_u32;
    let delimiter = "::";
    let mut harness =
        parser_with_properties(&[("max-record-size", &maximum), ("delimiter", &delimiter)]);
    assert_eq!(harness.push(buffer(b"abc:")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(harness.buffers_in_queue(), 0);
    assert_eq!(harness.push(buffer(b":")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"abc");
}

#[test]
fn lineparse_preserves_binary_payload() {
    let mut harness = parser();
    assert_eq!(
        harness.push(buffer(&[0xff, b'\n'])),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(pull_bytes(&mut harness), vec![0xff]);
}

#[test]
fn lineparse_flush_discards_incomplete_record() {
    let mut harness = parser();
    assert_eq!(
        harness.push(buffer(b"stale-without-a-delimiter")),
        Ok(gst::FlowSuccess::Ok)
    );
    assert!(harness.push_event(gst::event::FlushStart::new()));
    assert!(harness.push_event(gst::event::FlushStop::new(true)));
    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
    assert!(harness.push_event(gst::event::Segment::new(&segment)));
    assert_eq!(harness.push(buffer(b"\nnew\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"");
    assert_eq!(pull_bytes(&mut harness), b"new");
    assert_eq!(harness.buffers_in_queue(), 0);
}

#[test]
fn lineenc_appends_missing_default_delimiter_once() {
    let mut harness = encoder_with_delimiter(None);
    assert_eq!(harness.push(buffer(b"record")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"record\n");
}

#[test]
fn lineenc_property_defaults_and_round_trips_in_ready() {
    let encoder = element("lineenc");
    assert_eq!(encoder.property::<String>("delimiter"), "\n");
    assert_ne!(
        encoder.set_state(gst::State::Ready),
        Err(gst::StateChangeError)
    );
    encoder.set_property("delimiter", "::");
    assert_eq!(encoder.property::<String>("delimiter"), "::");
    encoder
        .set_state(gst::State::Null)
        .expect("returning encoder to NULL");
}

#[test]
fn lineenc_does_not_duplicate_existing_full_delimiter() {
    let mut harness = encoder_with_delimiter(None);
    assert_eq!(harness.push(buffer(b"record\n")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"record\n");
}

#[test]
fn lineenc_partial_suffix_receives_full_delimiter() {
    let mut harness = encoder_with_delimiter(Some("::"));
    assert_eq!(harness.push(buffer(b"record:")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"record:::");
}

#[test]
fn lineenc_custom_multibyte_delimiter_works() {
    let mut harness = encoder_with_delimiter(Some("<END>"));
    assert_eq!(harness.push(buffer(b"record")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"record<END>");
}

#[test]
fn lineenc_empty_buffer_becomes_delimiter() {
    let mut harness = encoder_with_delimiter(None);
    assert_eq!(harness.push(buffer(b"")), Ok(gst::FlowSuccess::Ok));
    assert_eq!(pull_bytes(&mut harness), b"\n");
}

#[test]
fn lineenc_preserves_binary_payload() {
    let mut harness = encoder_with_delimiter(None);
    assert_eq!(
        harness.push(buffer(&[0xff, 0x00])),
        Ok(gst::FlowSuccess::Ok)
    );
    assert_eq!(pull_bytes(&mut harness), vec![0xff, 0x00, b'\n']);
}

#[test]
fn lineenc_preserves_buffer_metadata() {
    let mut harness = encoder_with_delimiter(None);
    let mut input = buffer(b"metadata");
    {
        let input = input.get_mut().expect("writable input buffer");
        input.set_pts(gst::ClockTime::from_seconds(5));
        input.set_dts(gst::ClockTime::from_seconds(4));
        input.set_duration(gst::ClockTime::from_mseconds(250));
        input.set_offset(11);
        input.set_offset_end(19);
        input.set_flags(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER);
        let reference = gst::Caps::builder("timestamp/x-test").build();
        gst::ReferenceTimestampMeta::add(input, &reference, gst::ClockTime::from_seconds(42), None);
    }

    assert_eq!(harness.push(input), Ok(gst::FlowSuccess::Ok));
    let output = harness.pull().expect("pulling encoded buffer");
    assert_eq!(bytes(&output), b"metadata\n");
    assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(5)));
    assert_eq!(output.dts(), Some(gst::ClockTime::from_seconds(4)));
    assert_eq!(output.duration(), Some(gst::ClockTime::from_mseconds(250)));
    assert_eq!(output.offset(), 11);
    assert_eq!(output.offset_end(), 19);
    assert!(
        output
            .flags()
            .contains(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER)
    );
    assert!(output.meta::<gst::ReferenceTimestampMeta>().is_some());
}

#[test]
fn lineenc_empty_delimiter_fails_cleanly() {
    init();
    let encoder = gst::ElementFactory::make("lineenc")
        .property("delimiter", "")
        .build()
        .expect("constructing lineenc");
    assert_eq!(
        encoder.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    encoder
        .set_state(gst::State::Null)
        .expect("returning encoder to NULL after rejected state change");
}

#[test]
fn lines_round_trip_preserves_empty_and_binary_records() {
    init();
    let mut harness = gst_check::Harness::new_parse("lineenc ! lineparse");
    harness.set_src_caps_str("application/octet-stream");
    harness.play();

    for record in [b"alpha".as_slice(), b"".as_slice(), &[0xff, 0x00]] {
        assert_eq!(harness.push(buffer(record)), Ok(gst::FlowSuccess::Ok));
    }

    assert_eq!(pull_bytes(&mut harness), b"alpha");
    assert_eq!(pull_bytes(&mut harness), b"");
    assert_eq!(pull_bytes(&mut harness), vec![0xff, 0x00]);
    assert_eq!(harness.buffers_in_queue(), 0);
}
