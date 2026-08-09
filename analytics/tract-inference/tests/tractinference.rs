#![expect(
    clippy::expect_used,
    reason = "test setup and assertions require successful GStreamer operations"
)]

use std::fs;
use std::sync::Once;

use gst::prelude::*;

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gsttractinference::plugin_register_static().expect("registering Tract inference plugin");
    });
}

fn fixture_element() -> Result<(gst::Element, tempfile::TempDir), Box<dyn std::error::Error>> {
    init();
    let directory = tempfile::tempdir()?;
    let model = directory.path().join("identity.onnx");
    fs::write(
        &model,
        include_bytes!("../../inference-common/tests/fixtures/identity.onnx"),
    )?;
    fs::write(
        directory.path().join("identity.onnx.modelinfo"),
        include_str!("../../inference-common/tests/fixtures/identity.onnx.modelinfo"),
    )?;
    let element = gst::ElementFactory::make("tractinference")
        .property("model-file", model.to_string_lossy().as_ref())
        .build()?;
    Ok((element, directory))
}

fn video_caps() -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 2_i32)
        .field("height", 1_i32)
        .field("framerate", gst::Fraction::new(1, 1))
        .build()
}

fn input_buffer(caps: &gst::Caps) -> gst::Buffer {
    let info = gst_video::VideoInfo::from_caps(caps).expect("building video info");
    let mut buffer = gst::Buffer::with_size(info.size()).expect("allocating video buffer");
    let buffer_ref = buffer.get_mut().expect("writable video buffer");
    let mut map = buffer_ref.map_writable().expect("mapping video buffer");
    let pixels = map
        .as_mut_slice()
        .get_mut(..6)
        .expect("video buffer is large enough for the fixture frame");
    pixels.copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    drop(map);
    buffer
}

fn tensor_values(tensor: &gst_analytics::Tensor) -> Vec<f32> {
    tensor
        .data()
        .map_readable()
        .expect("mapping tensor data")
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 tensor chunk")))
        .collect()
}

#[test]
fn runs_fixture_and_preserves_video_buffer_metadata() {
    let (element, _directory) = fixture_element().expect("creating fixture element");
    let caps = video_caps();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();

    let mut input = input_buffer(&caps);
    let expected_video = input
        .map_readable()
        .expect("mapping expected video")
        .as_slice()
        .to_vec();
    {
        let input = input.get_mut().expect("writable test buffer");
        input.set_pts(gst::ClockTime::from_seconds(5));
        input.set_dts(gst::ClockTime::from_seconds(4));
        input.set_duration(gst::ClockTime::from_mseconds(250));
        input.set_flags(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER);
        let reference = gst::Caps::builder("timestamp/x-test").build();
        gst::ReferenceTimestampMeta::add(input, &reference, gst::ClockTime::from_seconds(42), None);
    }
    let output = harness.push_and_pull(input).expect("running fixture model");

    assert_eq!(
        output
            .map_readable()
            .expect("mapping output video")
            .as_slice(),
        expected_video,
        "in-place transform must preserve video bytes"
    );
    assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(5)));
    assert_eq!(output.dts(), Some(gst::ClockTime::from_seconds(4)));
    assert_eq!(output.duration(), Some(gst::ClockTime::from_mseconds(250)));
    assert!(
        output
            .flags()
            .contains(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER)
    );
    assert!(output.meta::<gst::ReferenceTimestampMeta>().is_some());

    let meta = output
        .meta::<gst_analytics::TensorMeta>()
        .expect("tensor metadata attached");
    let tensors = meta.as_slice();
    assert_eq!(tensors.len(), 2);
    for (tensor, id) in tensors.iter().zip(["first", "second"]) {
        assert_eq!(tensor.id(), gst::glib::Quark::from_str(id));
        assert_eq!(tensor.data_type(), gst_analytics::TensorDataType::Float32);
        assert_eq!(tensor.dims(), [1, 1, 2, 3]);
        assert_eq!(tensor.dims_order(), gst_analytics::TensorDimOrder::RowMajor);
        assert_eq!(tensor_values(tensor), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    let negotiated = element
        .static_pad("src")
        .expect("source pad exists")
        .current_caps()
        .expect("source caps negotiated");
    let groups = negotiated
        .structure(0)
        .expect("source caps structure exists")
        .get::<gst::Structure>("tensors")
        .expect("tensor groups advertised");
    let fixture_group = groups
        .get::<gst::UniqueList>("gstsmith-identity-fixture")
        .expect("tensor group uses the GStreamer 1.28 unique-list contract");
    assert_eq!(fixture_group.as_slice().len(), 2);
}

#[test]
fn rejects_model_size_during_negotiation() {
    let (element, _directory) = fixture_element().expect("creating fixture element");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(
        gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", 3_i32)
            .field("height", 1_i32)
            .field("framerate", gst::Fraction::new(1, 1))
            .build(),
    );
    harness.play();
    assert!(
        harness
            .push(gst::Buffer::with_size(24).expect("allocating mismatched video buffer"))
            .is_err(),
        "model/video size mismatch was accepted"
    );
}
