use std::str::FromStr;
use std::sync::Once;

use gst::prelude::*;

#[expect(
    clippy::expect_used,
    reason = "all tests require successful one-time GStreamer and plugin initialization"
)]
fn init() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstconsole::plugin_register_static().expect("registering the console plugin");
    });
}

#[test]
fn registers_all_elements() {
    init();

    assert!(gst::ElementFactory::find("consolesrc").is_some());
    assert!(gst::ElementFactory::find("consolesink").is_some());
    assert!(gst::ElementFactory::find("consoleprint").is_some());
}

#[test]
fn transport_elements_expose_any_caps() {
    init();

    for (factory, pad_name) in [("consolesrc", "src"), ("consolesink", "sink")] {
        let element = gst::ElementFactory::make(factory)
            .build()
            .expect("constructing console element");
        let pad = element
            .static_pad(pad_name)
            .expect("finding the console element pad");
        let caps = pad.pad_template_caps();

        assert!(
            caps.is_any(),
            "{factory}:{pad_name} does not expose ANY caps"
        );
    }
}

#[test]
fn consoleprint_exposes_text_and_json_caps() {
    init();

    for pad_name in ["sink", "src"] {
        let element = gst::ElementFactory::make("consoleprint")
            .build()
            .expect("constructing consoleprint");
        let pad = element
            .static_pad(pad_name)
            .expect("finding the consoleprint pad");
        let caps = pad.pad_template_caps();

        for expected in [
            "text/x-raw, format=utf8",
            "application/json",
            "application/x-json",
        ] {
            let expected = gst::Caps::from_str(expected).expect("valid test caps");
            assert!(
                caps.can_intersect(&expected),
                "consoleprint:{pad_name} is missing caps {expected}"
            );
        }

        let binary = gst::Caps::from_str("application/octet-stream").expect("valid test caps");
        assert!(
            !caps.can_intersect(&binary),
            "consoleprint:{pad_name} accepts binary caps"
        );
    }
}

#[test]
fn consolesrc_reads_standard_input_through_core_elements() {
    init();

    let source = gst::ElementFactory::make("consolesrc")
        .build()
        .expect("constructing consolesrc")
        .downcast::<gst::Bin>()
        .expect("consolesrc is a bin");
    let stdin = source.by_name("stdin").expect("finding the stdin source");

    assert!(
        stdin
            .factory()
            .is_some_and(|factory| factory.name() == "fdsrc")
    );
    assert_eq!(stdin.property::<i32>("fd"), 0);
    assert!(source.by_name("caps").is_none());
}

#[test]
fn consolesink_exposes_only_stream_property() {
    init();

    let element = gst::ElementFactory::make("consolesink")
        .build()
        .expect("constructing consolesink");

    assert!(element.find_property("stream").is_some());
    assert!(element.find_property("ensure-newline").is_none());
    assert_eq!(
        element.property::<gstconsole::ConsoleStream>("stream"),
        gstconsole::ConsoleStream::Stdout
    );

    element.set_property("stream", gstconsole::ConsoleStream::Stderr);

    assert_eq!(
        element.property::<gstconsole::ConsoleStream>("stream"),
        gstconsole::ConsoleStream::Stderr
    );
}

#[test]
fn consoleprint_retains_text_output_properties() {
    init();

    let element = gst::ElementFactory::make("consoleprint")
        .build()
        .expect("constructing consoleprint");

    assert!(element.find_property("stream").is_some());
    assert!(element.find_property("ensure-newline").is_some());
    assert_eq!(
        element.property::<gstconsole::ConsoleStream>("stream"),
        gstconsole::ConsoleStream::Stdout
    );
    assert!(element.property::<bool>("ensure-newline"));

    element.set_property("stream", gstconsole::ConsoleStream::Stderr);
    element.set_property("ensure-newline", false);

    assert_eq!(
        element.property::<gstconsole::ConsoleStream>("stream"),
        gstconsole::ConsoleStream::Stderr
    );
    assert!(!element.property::<bool>("ensure-newline"));
}

#[test]
fn console_transport_elements_link_to_core_elements() {
    init();

    let fakesrc = gst::ElementFactory::make("fakesrc")
        .build()
        .expect("constructing fakesrc");
    let sink = gst::ElementFactory::make("consolesink")
        .build()
        .expect("constructing consolesink");
    fakesrc.link(&sink).expect("linking fakesrc to consolesink");

    let source = gst::ElementFactory::make("consolesrc")
        .build()
        .expect("constructing consolesrc");
    let fakesink = gst::ElementFactory::make("fakesink")
        .build()
        .expect("constructing fakesink");
    source
        .link(&fakesink)
        .expect("linking consolesrc to fakesink");
}

#[test]
fn consoleprint_passes_buffers_through_unchanged() {
    init();

    let element = gst::ElementFactory::make("consoleprint")
        .property("ensure-newline", false)
        .build()
        .expect("constructing consoleprint");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps_str("text/x-raw, format=utf8");

    let mut input = gst::Buffer::from_mut_slice(Vec::<u8>::new());
    input
        .get_mut()
        .expect("writable test buffer")
        .set_pts(gst::ClockTime::from_seconds(3));

    assert_eq!(harness.push(input), Ok(gst::FlowSuccess::Ok));

    let output = harness.pull().expect("pulling pass-through buffer");
    assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(3)));
    assert_eq!(
        output
            .map_readable()
            .expect("mapping pass-through buffer")
            .as_slice(),
        b""
    );
}

#[test]
fn consolesink_accepts_an_empty_binary_buffer() {
    init();

    let element = gst::ElementFactory::make("consolesink")
        .build()
        .expect("constructing consolesink");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), None);
    harness.set_src_caps_str("application/octet-stream");

    assert_eq!(
        harness.push(gst::Buffer::from_mut_slice(Vec::<u8>::new())),
        Ok(gst::FlowSuccess::Ok)
    );
}

#[test]
fn rejects_invalid_utf8() {
    init();

    let element = gst::ElementFactory::make("consoleprint")
        .property("ensure-newline", false)
        .build()
        .expect("constructing consoleprint");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps_str("text/x-raw, format=utf8");

    assert_eq!(
        harness.push(gst::Buffer::from_mut_slice(vec![0xff])),
        Err(gst::FlowError::Error)
    );
}
