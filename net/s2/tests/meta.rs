mod common;

use gst::prelude::*;

fn header(name: Vec<u8>, value: Vec<u8>) -> gst::glib::SendValue {
    gst::Structure::builder("s2-header")
        .field("name", gst::glib::Bytes::from_owned(name))
        .field("value", gst::glib::Bytes::from_owned(value))
        .build()
        .to_send_value()
}

#[test]
fn binary_duplicate_headers_and_zero_values_survive_buffer_copy() {
    common::init();
    let _source = common::element("s2src");
    let mut buffer = gst::Buffer::new();
    let mut meta = gst::meta::CustomMeta::add(
        buffer.get_mut().expect("new buffer is writable"),
        "GstS2RecordMeta",
    )
    .expect("adding S2 metadata");
    let structure = meta.mut_structure();
    structure.set("basin", "test-basin");
    structure.set("stream", "stream");
    structure.set("seq-num", 0_u64);
    structure.set("timestamp", 0_u64);
    structure.set("is-command", false);
    structure.set(
        "headers",
        gst::Array::new([
            header(vec![0, 255], vec![1, 0]),
            header(vec![0, 255], Vec::new()),
        ]),
    );

    let copied = buffer.copy();
    let copied_meta = gst::meta::CustomMeta::from_buffer(&copied, "GstS2RecordMeta")
        .expect("metadata survives buffer copy");
    let copied_structure = copied_meta.structure();
    assert_eq!(
        copied_structure
            .get::<String>("basin")
            .expect("metadata basin"),
        "test-basin"
    );
    assert_eq!(
        copied_structure
            .get::<u64>("seq-num")
            .expect("metadata sequence"),
        0
    );
    let headers = copied_structure
        .get::<gst::Array>("headers")
        .expect("metadata headers");
    assert_eq!(headers.len(), 2);
    let first = headers
        .iter()
        .next()
        .expect("first header")
        .get::<gst::Structure>()
        .expect("header structure");
    assert_eq!(
        first
            .get::<gst::glib::Bytes>("name")
            .expect("binary header name")
            .as_ref(),
        &[0, 255]
    );
}

#[test]
fn command_header_shape_survives_copy() {
    common::init();
    let mut buffer = gst::Buffer::new();
    let mut meta = gst::meta::CustomMeta::add(
        buffer.get_mut().expect("new buffer is writable"),
        "GstS2RecordMeta",
    )
    .expect("adding S2 metadata");
    let structure = meta.mut_structure();
    structure.set("basin", "test-basin");
    structure.set("stream", "stream");
    structure.set("seq-num", u64::MAX);
    structure.set("timestamp", u64::MAX);
    structure.set("is-command", true);
    structure.set(
        "headers",
        gst::Array::new([header(Vec::new(), b"fence".to_vec())]),
    );
    let copied = buffer.copy();
    let copied_meta = gst::meta::CustomMeta::from_buffer(&copied, "GstS2RecordMeta")
        .expect("metadata survives copy");
    assert!(
        copied_meta
            .structure()
            .get::<bool>("is-command")
            .expect("command marker")
    );
}
