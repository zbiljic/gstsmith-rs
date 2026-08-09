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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_fixture_element(provider: &str) -> gst::Element {
    let model =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metal-conv.onnx");
    gst::ElementFactory::make("tractinference")
        .property("model-file", model.to_string_lossy().as_ref())
        .property_from_str("execution-provider", provider)
        .build()
        .expect("creating convolution fixture element")
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_fixture_caps() -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 2_i32)
        .field("height", 2_i32)
        .field("framerate", gst::Fraction::new(1, 1))
        .build()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal_fixture(
    provider: &str,
) -> (String, gst_analytics::TensorDataType, Vec<usize>, Vec<f32>) {
    let element = metal_fixture_element(provider);
    let caps = metal_fixture_caps();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();
    let output = harness
        .push_and_pull(input_buffer(&caps))
        .expect("running convolution fixture through the GStreamer harness");
    let meta = output
        .meta::<gst_analytics::TensorMeta>()
        .expect("convolution tensor metadata attached");
    let tensor = meta
        .as_slice()
        .first()
        .expect("convolution fixture emitted one tensor");
    (
        tensor.id().as_str().to_string(),
        tensor.data_type(),
        tensor.dims().to_vec(),
        tensor_values(tensor),
    )
}

fn input_buffer(caps: &gst::Caps) -> gst::Buffer {
    let info = gst_video::VideoInfo::from_caps(caps).expect("building video info");
    let mut buffer = gst::Buffer::with_size(info.size()).expect("allocating video buffer");
    let buffer_ref = buffer.get_mut().expect("writable video buffer");
    let mut map = buffer_ref.map_writable().expect("mapping video buffer");
    let stride = usize::try_from(
        *info
            .stride()
            .first()
            .expect("fixture video info has a plane stride"),
    )
    .expect("positive fixture stride");
    let row_bytes = usize::try_from(info.width()).expect("fixture width fits usize") * 3;
    let mut value = 1_u8;
    for row in map
        .as_mut_slice()
        .chunks_mut(stride)
        .take(usize::try_from(info.height()).expect("fixture height fits usize"))
    {
        for byte in row
            .get_mut(..row_bytes)
            .expect("video row is large enough for fixture pixels")
        {
            *byte = value;
            value = value.saturating_add(1);
        }
    }
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

fn execution_provider_nick(element: &gst::Element) -> String {
    let value = element.property_value("execution-provider");
    let (_class, enum_value) = gst::glib::EnumValue::from_value(&value)
        .expect("execution-provider property has an enum value");
    enum_value.nick().to_owned()
}

#[test]
fn execution_provider_has_stable_values_and_round_trips() {
    init();
    let element = gst::ElementFactory::make("tractinference")
        .build()
        .expect("creating Tract inference element");
    let pspec = element
        .find_property("execution-provider")
        .expect("execution-provider property exists");
    assert_eq!(pspec.value_type().name(), "GstSmithTractExecutionProvider");
    let class = gst::glib::EnumClass::with_type(pspec.value_type())
        .expect("execution-provider uses a GLib enum");
    assert_eq!(class.value(0).map(gst::glib::EnumValue::nick), Some("cpu"));
    assert_eq!(
        class.value(1).map(gst::glib::EnumValue::nick),
        Some("metal")
    );
    assert_eq!(execution_provider_nick(&element), "cpu");
    element.set_property_from_str("execution-provider", "metal");
    assert_eq!(execution_provider_nick(&element), "metal");
    element.set_property_from_str("execution-provider", "cpu");
    assert_eq!(execution_provider_nick(&element), "cpu");
}

#[cfg(any(not(target_os = "macos"), not(feature = "metal")))]
#[test]
fn unavailable_metal_provider_fails_without_cpu_fallback() {
    init();
    let element = gst::ElementFactory::make("tractinference")
        .property_from_str("execution-provider", "metal")
        .build()
        .expect("creating Metal-configured element");
    let pipeline = gst::Pipeline::new();
    pipeline.add(&element).expect("adding element to pipeline");
    assert!(
        pipeline.set_state(gst::State::Paused).is_err(),
        "unavailable Metal provider unexpectedly started"
    );
    let message = pipeline
        .bus()
        .expect("pipeline bus exists")
        .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error])
        .expect("provider settings error was posted");
    assert!(
        matches!(message.view(), gst::MessageView::Error(_)),
        "filtered message was not an error"
    );
    let gst::MessageView::Error(error) = message.view() else {
        return;
    };
    let details = format!("{} {:?}", error.error(), error.debug());
    #[cfg(not(target_os = "macos"))]
    assert!(details.contains("only supported on macOS"), "{details}");
    #[cfg(all(target_os = "macos", not(feature = "metal")))]
    assert!(
        details.contains("not compiled") && details.contains("metal"),
        "{details}"
    );
    pipeline
        .set_state(gst::State::Null)
        .expect("stopping failed pipeline");
}

// `metal-conv.onnx` was generated with ONNX's `onnx.proto3` and
// `protoc --encode=onnx.ModelProto`: one [1,3,2,2] float input feeds a 1x1 Conv
// with weights [1,2,3] and bias 0.5, producing one [1,1,2,2] float output.
// For RGB bytes 1 through 12, preprocessing and the convolution deterministically
// produce [14.5, 32.5, 50.5, 68.5].
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn cpu_and_metal_providers_preserve_the_tensor_contract() {
    init();
    let cpu = run_metal_fixture("cpu");
    let metal = run_metal_fixture("metal");
    assert_eq!(cpu.0, "convolution");
    assert_eq!(cpu.0, metal.0);
    assert_eq!(cpu.1, gst_analytics::TensorDataType::Float32);
    assert_eq!(cpu.1, metal.1);
    assert_eq!(cpu.2, [1, 1, 2, 2]);
    assert_eq!(cpu.2, metal.2);
    for (actual, expected) in cpu.3.iter().zip([14.5, 32.5, 50.5, 68.5]) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }
    assert_eq!(cpu.3.len(), metal.3.len());
    for (cpu_value, metal_value) in cpu.3.iter().zip(&metal.3) {
        assert!(
            (cpu_value - metal_value).abs() <= 1.0e-5,
            "CPU value {cpu_value} differs from Metal value {metal_value}"
        );
    }
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
