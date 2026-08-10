#![expect(
    clippy::expect_used,
    reason = "GStreamer harness setup failures should fail the integration test"
)]
#![expect(
    clippy::panic_in_result_fn,
    clippy::unwrap_in_result,
    reason = "assertions keep parity failures concise in Result-returning integration tests"
)]

use std::fs;
use std::sync::Once;
use std::time::Instant;

use gst::glib::prelude::*;
use gst::prelude::*;

const MODEL_INFO: &str =
    include_str!("../../inference-common/tests/fixtures/identity.onnx.modelinfo");
const MODEL: &[u8] = include_bytes!("../../inference-common/tests/fixtures/identity.onnx");

// Generated from a minimal ONNX textproto with the repository's Apache-2.0
// licensed `onnx.proto3` and `protoc --encode=onnx.ModelProto`. A supported
// 1x1 Conv feeds Erf, which pinned CoreML does not support. This keeps both a
// CoreML partition and a deterministic CPU-fallback node in the same graph.
#[cfg(all(feature = "coreml", target_os = "macos"))]
#[rustfmt::skip]
const COREML_PARTIAL_MODEL: &[u8] = &[
    0x08, 0x09, 0x12, 0x1a, 0x67, 0x73, 0x74, 0x73, 0x6d, 0x69, 0x74, 0x68, 0x20, 0x66, 0x69, 0x78,
    0x74, 0x75, 0x72, 0x65, 0x20, 0x67, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x6f, 0x72, 0x3a, 0xd9,
    0x01, 0x0a, 0x2f, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x0a, 0x07, 0x77, 0x65, 0x69,
    0x67, 0x68, 0x74, 0x73, 0x0a, 0x04, 0x62, 0x69, 0x61, 0x73, 0x12, 0x0b, 0x63, 0x6f, 0x6e,
    0x76, 0x6f, 0x6c, 0x75, 0x74, 0x69, 0x6f, 0x6e, 0x1a, 0x04, 0x63, 0x6f, 0x6e, 0x76, 0x22,
    0x04, 0x43, 0x6f, 0x6e, 0x76, 0x0a, 0x1f, 0x0a, 0x0b, 0x63, 0x6f, 0x6e, 0x76, 0x6f, 0x6c,
    0x75, 0x74, 0x69, 0x6f, 0x6e, 0x12, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x1a, 0x03,
    0x65, 0x72, 0x66, 0x22, 0x03, 0x45, 0x72, 0x66, 0x12, 0x0e, 0x63, 0x6f, 0x72, 0x65, 0x6d,
    0x6c, 0x2d, 0x70, 0x61, 0x72, 0x74, 0x69, 0x61, 0x6c, 0x2a, 0x1f, 0x0a, 0x04, 0x01, 0x03,
    0x01, 0x01, 0x10, 0x01, 0x22, 0x0c, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00,
    0x00, 0x40, 0x40, 0x42, 0x07, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x2a, 0x11, 0x0a,
    0x01, 0x01, 0x10, 0x01, 0x22, 0x04, 0x00, 0x00, 0x00, 0x3f, 0x42, 0x04, 0x62, 0x69, 0x61,
    0x73, 0x5a, 0x1f, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x16, 0x0a, 0x14, 0x08,
    0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x02,
    0x0a, 0x02, 0x08, 0x02, 0x62, 0x20, 0x0a, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12,
    0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01,
    0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x02, 0x10, 0x0d,
];

#[cfg(all(feature = "coreml", target_os = "macos"))]
const COREML_PARTIAL_MODEL_INFO: &str = "[modelinfo]\n\
version=1.0\n\
group-id=gstsmith-coreml-partial-fixture\n\
\n\
[input]\n\
id=image\n\
type=float32\n\
dims=1,3,2,2\n\
dir=input\n\
ranges=0.0,255.0\n\
\n\
[output]\n\
id=result\n\
type=float32\n\
dims=1,1,2,2\n\
dir=output\n";

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstortinference::plugin_register_static().expect("registering ORT inference plugin");
        gsttractinference::plugin_register_static().expect("registering Tract inference plugin");
    });
}

fn fixture_element(
    factory: &str,
) -> Result<(gst::Element, tempfile::TempDir), Box<dyn std::error::Error>> {
    init();
    let directory = tempfile::tempdir()?;
    let model = directory.path().join("identity.onnx");
    fs::write(&model, MODEL)?;
    fs::write(directory.path().join("identity.onnx.modelinfo"), MODEL_INFO)?;
    let element = gst::ElementFactory::make(factory)
        .property("model-file", model.to_string_lossy().as_ref())
        .build()?;
    Ok((element, directory))
}

fn caps() -> gst::Caps {
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
    map.as_mut_slice()
        .get_mut(..6)
        .expect("fixture video has six pixel bytes")
        .copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    drop(map);
    buffer_ref.set_pts(gst::ClockTime::from_seconds(5));
    buffer_ref.set_dts(gst::ClockTime::from_seconds(4));
    buffer_ref.set_duration(gst::ClockTime::from_mseconds(250));
    buffer_ref.set_flags(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER);
    let reference = gst::Caps::builder("timestamp/x-test").build();
    gst::ReferenceTimestampMeta::add(
        buffer_ref,
        &reference,
        gst::ClockTime::from_seconds(42),
        None,
    );
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

fn enum_nick(element: &gst::Element, property: &str) -> String {
    let value = element.property_value(property);
    let (_class, enum_value) =
        gst::glib::EnumValue::from_value(&value).expect("enum property value");
    enum_value.nick().to_owned()
}

fn assert_tensor_contract(
    element: &gst::Element,
    output: &gst::Buffer,
    expected_values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let meta = output
        .meta::<gst_analytics::TensorMeta>()
        .ok_or_else(|| std::io::Error::other("tensor metadata was not attached"))?;
    let tensors = meta.as_slice();
    assert_eq!(tensors.len(), 2);
    for (tensor, id) in tensors.iter().zip(["first", "second"]) {
        assert_eq!(tensor.id(), gst::glib::Quark::from_str(id));
        assert_eq!(tensor.data_type(), gst_analytics::TensorDataType::Float32);
        assert_eq!(tensor.dims(), [1, 1, 2, 3]);
        assert_eq!(tensor.dims_order(), gst_analytics::TensorDimOrder::RowMajor);
        assert_eq!(tensor_values(tensor), expected_values);
    }

    let negotiated = element
        .static_pad("src")
        .ok_or_else(|| std::io::Error::other("source pad is missing"))?
        .current_caps()
        .ok_or_else(|| std::io::Error::other("source caps were not negotiated"))?;
    let structure = negotiated
        .structure(0)
        .ok_or_else(|| std::io::Error::other("source caps structure is missing"))?;
    assert_eq!(structure.get::<String>("format")?, "RGB");
    let groups = structure.get::<gst::Structure>("tensors")?;
    let descriptors = groups.get::<gst::UniqueList>("gstsmith-identity-fixture")?;
    assert_eq!(descriptors.as_slice().len(), 2);
    for (descriptor, id) in descriptors.as_slice().iter().zip(["first", "second"]) {
        let descriptor = descriptor.get::<gst::Caps>()?;
        let descriptor = descriptor
            .structure(0)
            .ok_or_else(|| std::io::Error::other("tensor descriptor has no structure"))?;
        assert_eq!(descriptor.get::<String>("tensor-id")?, id);
        assert_eq!(descriptor.get::<String>("type")?, "float32");
        assert_eq!(descriptor.get::<String>("dims-order")?, "row-major");
        let dims = descriptor.get::<gst::Array>("dims")?;
        let dims = dims
            .iter()
            .map(|value| value.get::<i32>())
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(dims, [1, 1, 2, 3]);
    }
    Ok(())
}

fn run_factory(
    factory: &str,
) -> Result<(gst::Buffer, gst::Caps, tempfile::TempDir), Box<dyn std::error::Error>> {
    run_factory_with_order(factory, "rgb", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

fn run_factory_with_order(
    factory: &str,
    order: &str,
    expected_values: &[f32],
) -> Result<(gst::Buffer, gst::Caps, tempfile::TempDir), Box<dyn std::error::Error>> {
    let (element, directory) = fixture_element(factory)?;
    element.set_property_from_str("model-channel-order", order);
    let caps = caps();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();
    let output = harness.push_and_pull(input_buffer(&caps))?;
    assert_tensor_contract(&element, &output, expected_values)?;
    Ok((output, caps, directory))
}

#[test]
fn bgr_model_order_matches_backends_and_preserves_truthful_rgb_video()
-> Result<(), Box<dyn std::error::Error>> {
    let expected_tensors = [3.0, 2.0, 1.0, 6.0, 5.0, 4.0];
    let (ort_output, caps, _ort_directory) =
        run_factory_with_order("ortinference", "bgr", &expected_tensors)?;
    let (tract_output, _tract_caps, _tract_directory) =
        run_factory_with_order("tractinference", "bgr", &expected_tensors)?;
    let expected_video = input_buffer(&caps).map_readable()?.as_slice().to_vec();
    for output in [&ort_output, &tract_output] {
        assert_eq!(output.map_readable()?.as_slice(), expected_video);
        assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(5)));
        assert_eq!(output.dts(), Some(gst::ClockTime::from_seconds(4)));
        assert_eq!(output.duration(), Some(gst::ClockTime::from_mseconds(250)));
        assert!(
            output
                .flags()
                .contains(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER)
        );
        assert!(output.meta::<gst::ReferenceTimestampMeta>().is_some());
    }
    let ort_meta = ort_output
        .meta::<gst_analytics::TensorMeta>()
        .expect("ORT metadata");
    let tract_meta = tract_output
        .meta::<gst_analytics::TensorMeta>()
        .expect("Tract metadata");
    for (ort_tensor, tract_tensor) in ort_meta.as_slice().iter().zip(tract_meta.as_slice()) {
        assert_eq!(ort_tensor.id(), tract_tensor.id());
        assert_eq!(ort_tensor.data_type(), tract_tensor.data_type());
        assert_eq!(ort_tensor.dims(), tract_tensor.dims());
        assert_eq!(ort_tensor.dims_order(), tract_tensor.dims_order());
        assert_eq!(tensor_values(ort_tensor), tensor_values(tract_tensor));
    }
    Ok(())
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn run_coreml_fixture(
    strict: bool,
) -> Result<(gst::Buffer, tempfile::TempDir), Box<dyn std::error::Error>> {
    init();
    let directory = tempfile::tempdir()?;
    let model = directory.path().join("coreml-conv.onnx");
    fs::write(
        &model,
        include_bytes!("../../tract-inference/tests/fixtures/metal-conv.onnx"),
    )?;
    fs::write(
        directory.path().join("coreml-conv.onnx.modelinfo"),
        include_str!("../../tract-inference/tests/fixtures/metal-conv.onnx.modelinfo"),
    )?;
    let element = gst::ElementFactory::make("ortinference")
        .property("model-file", model.to_string_lossy().as_ref())
        .property_from_str("execution-provider", "coreml")
        .property("strict-execution-provider", strict)
        .build()?;
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 2_i32)
        .field("height", 2_i32)
        .field("framerate", gst::Fraction::new(1, 1))
        .build();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();
    let output = harness.push_and_pull(input_buffer(&caps))?;
    let tensors = output
        .meta::<gst_analytics::TensorMeta>()
        .ok_or_else(|| std::io::Error::other("CoreML output tensor metadata is missing"))?;
    assert_eq!(tensors.as_slice().len(), 1);
    let tensor = tensors
        .as_slice()
        .first()
        .ok_or_else(|| std::io::Error::other("CoreML output tensor is missing"))?;
    assert_eq!(tensor.dims(), [1, 1, 2, 2]);
    Ok((output, directory))
}

#[test]
fn ort_and_tract_factories_have_identical_fixture_contract_and_video_passthrough()
-> Result<(), Box<dyn std::error::Error>> {
    let (ort_output, caps, _ort_directory) = run_factory("ortinference")?;
    let (tract_output, _tract_caps, _tract_directory) = run_factory("tractinference")?;
    let expected_video = input_buffer(&caps)
        .map_readable()
        .expect("mapping expected video")
        .as_slice()
        .to_vec();
    for output in [&ort_output, &tract_output] {
        assert_eq!(output.map_readable()?.as_slice(), expected_video);
        assert_eq!(output.pts(), Some(gst::ClockTime::from_seconds(5)));
        assert_eq!(output.dts(), Some(gst::ClockTime::from_seconds(4)));
        assert_eq!(output.duration(), Some(gst::ClockTime::from_mseconds(250)));
        assert!(
            output
                .flags()
                .contains(gst::BufferFlags::DISCONT | gst::BufferFlags::MARKER)
        );
        assert!(output.meta::<gst::ReferenceTimestampMeta>().is_some());
    }
    let ort_meta = ort_output
        .meta::<gst_analytics::TensorMeta>()
        .expect("ORT metadata");
    let tract_meta = tract_output
        .meta::<gst_analytics::TensorMeta>()
        .expect("Tract metadata");
    for (ort_tensor, tract_tensor) in ort_meta.as_slice().iter().zip(tract_meta.as_slice()) {
        assert_eq!(ort_tensor.id(), tract_tensor.id());
        assert_eq!(ort_tensor.data_type(), tract_tensor.data_type());
        assert_eq!(ort_tensor.dims(), tract_tensor.dims());
        assert_eq!(ort_tensor.dims_order(), tract_tensor.dims_order());
        for (ort_value, tract_value) in tensor_values(ort_tensor)
            .iter()
            .zip(tensor_values(tract_tensor))
        {
            assert!((ort_value - tract_value).abs() <= 1.0e-5);
        }
    }
    Ok(())
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_strict_supported_graph_runs() -> Result<(), Box<dyn std::error::Error>> {
    let (_output, _directory) = run_coreml_fixture(true)?;
    Ok(())
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn partial_coreml_element(
    strict: bool,
) -> Result<(gst::Element, tempfile::TempDir), Box<dyn std::error::Error>> {
    init();
    let directory = tempfile::tempdir()?;
    let model = directory.path().join("coreml-partial.onnx");
    fs::write(&model, COREML_PARTIAL_MODEL)?;
    fs::write(
        directory.path().join("coreml-partial.onnx.modelinfo"),
        COREML_PARTIAL_MODEL_INFO,
    )?;
    let element = gst::ElementFactory::make("ortinference")
        .property("model-file", model.to_string_lossy().as_ref())
        .property_from_str("execution-provider", "coreml")
        .property("strict-execution-provider", strict)
        .build()?;
    Ok((element, directory))
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_partial_graph_uses_default_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let (element, _directory) = partial_coreml_element(false)?;
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 2_i32)
        .field("height", 2_i32)
        .field("framerate", gst::Fraction::new(1, 1))
        .build();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();
    let output = harness.push_and_pull(input_buffer(&caps))?;
    assert!(output.meta::<gst_analytics::TensorMeta>().is_some());
    Ok(())
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_partial_graph_fails_when_cpu_fallback_is_disabled()
-> Result<(), Box<dyn std::error::Error>> {
    let (element, _directory) = partial_coreml_element(true)?;
    let pipeline = gst::Pipeline::new();
    pipeline.add(&element)?;
    let _startup_error = pipeline
        .set_state(gst::State::Paused)
        .expect_err("strict partial CoreML graph must fail startup");
    let message = pipeline
        .bus()
        .ok_or_else(|| std::io::Error::other("pipeline bus is missing"))?
        .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error])
        .ok_or_else(|| std::io::Error::other("strict CoreML startup error is missing"))?;
    let gst::MessageView::Error(error) = message.view() else {
        return Err(std::io::Error::other("filtered message was not an error").into());
    };
    assert!(error.error().matches(gst::LibraryError::Settings));
    let details = format!("{} {:?}", error.error(), error.debug());
    assert!(details.contains("failed to load ONNX model"), "{details}");
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}

#[test]
fn properties_have_backend_defaults_and_ready_mutability() -> Result<(), Box<dyn std::error::Error>>
{
    init();
    let element = gst::ElementFactory::make("ortinference").build()?;
    assert_eq!(enum_nick(&element, "execution-provider"), "cpu");
    assert_eq!(element.property::<u32>("intra-op-threads"), 0);
    assert_eq!(enum_nick(&element, "graph-optimization"), "level3");
    assert_eq!(enum_nick(&element, "model-channel-order"), "rgb");
    assert!(!element.property::<bool>("strict-execution-provider"));
    element.set_property("strict-execution-provider", true);
    element.set_property_from_str("model-channel-order", "bgr");
    assert!(element.property::<bool>("strict-execution-provider"));
    assert_eq!(enum_nick(&element, "model-channel-order"), "bgr");
    element.set_property_from_str("model-channel-order", "rgb");
    assert_eq!(enum_nick(&element, "model-channel-order"), "rgb");
    for name in [
        "model-file",
        "model-info-file",
        "execution-provider",
        "intra-op-threads",
        "graph-optimization",
        "strict-execution-provider",
        "model-channel-order",
    ] {
        let property = element
            .find_property(name)
            .ok_or_else(|| std::io::Error::other("missing property"))?;
        assert!(
            property.flags().contains(gst::PARAM_FLAG_MUTABLE_READY),
            "{name} must be READY-mutable"
        );
    }
    let property = element
        .find_property("model-channel-order")
        .ok_or_else(|| std::io::Error::other("missing model channel order property"))?;
    assert_eq!(property.value_type().name(), "GstSmithOrtModelChannelOrder");
    let class = gst::glib::EnumClass::with_type(property.value_type())
        .ok_or_else(|| std::io::Error::other("model channel order is not an enum"))?;
    assert_eq!(class.value(0).map(gst::glib::EnumValue::nick), Some("rgb"));
    assert_eq!(class.value(1).map(gst::glib::EnumValue::nick), Some("bgr"));
    Ok(())
}

fn assert_cpu_strict_rejected() -> Result<(), Box<dyn std::error::Error>> {
    init();
    let element = gst::ElementFactory::make("ortinference")
        .property("strict-execution-provider", true)
        .build()?;
    let pipeline = gst::Pipeline::new();
    pipeline.add(&element)?;
    let _startup_error = pipeline
        .set_state(gst::State::Paused)
        .expect_err("CPU strict mode must fail startup");
    let message = pipeline
        .bus()
        .ok_or_else(|| std::io::Error::other("pipeline bus is missing"))?
        .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error])
        .ok_or_else(|| std::io::Error::other("settings error message is missing"))?;
    let gst::MessageView::Error(error) = message.view() else {
        return Err(std::io::Error::other("filtered message was not an error").into());
    };
    assert!(error.error().matches(gst::LibraryError::Settings));
    let details = format!("{} {:?}", error.error(), error.debug());
    assert!(details.contains("strict-execution-provider"), "{details}");
    assert!(details.contains("execution-provider=cpu"), "{details}");
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}

#[test]
fn strict_execution_provider_rejects_cpu_before_model_loading()
-> Result<(), Box<dyn std::error::Error>> {
    assert_cpu_strict_rejected()
}

#[test]
fn both_backend_registrations_are_isolated_and_resolve_separate_factories()
-> Result<(), Box<dyn std::error::Error>> {
    init();
    for factory in ["ortinference", "tractinference"] {
        let element = gst::ElementFactory::make(factory).build()?;
        assert_eq!(
            element.factory().map(|factory| factory.name().to_string()),
            Some(factory.to_owned())
        );
    }
    Ok(())
}

#[test]
fn malformed_model_info_fails_startup_instead_of_running() -> Result<(), Box<dyn std::error::Error>>
{
    let (element, directory) = fixture_element("ortinference")?;
    let invalid = directory.path().join("invalid.modelinfo");
    fs::write(&invalid, MODEL_INFO.replace("[x]", "[wrong-input]"))?;
    element.set_property("model-info-file", invalid.to_string_lossy().as_ref());
    let pipeline = gst::Pipeline::new();
    pipeline.add(&element)?;
    let _startup_error = pipeline
        .set_state(gst::State::Paused)
        .expect_err("invalid model info must fail startup");
    let message = pipeline
        .bus()
        .ok_or_else(|| std::io::Error::other("pipeline bus is missing"))?
        .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Error])
        .ok_or_else(|| std::io::Error::other("startup error message is missing"))?;
    assert!(matches!(message.view(), gst::MessageView::Error(_)));
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}

#[cfg(all(feature = "coreml", not(target_os = "macos")))]
#[test]
fn unavailable_coreml_fails_without_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let (element, _directory) = fixture_element("ortinference")?;
    element.set_property_from_str("execution-provider", "coreml");
    assert!(element.set_state(gst::State::Ready).is_err());
    element.set_state(gst::State::Null)?;
    Ok(())
}

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

#[test]
#[ignore = "run explicitly with GSTSMITH_BENCH_WARMUP and GSTSMITH_BENCH_ITERATIONS"]
fn benchmark_fixture_backends_reports_preprocessing_inference_and_total()
-> Result<(), Box<dyn std::error::Error>> {
    init();
    let warmup = env_count("GSTSMITH_BENCH_WARMUP", 10);
    let iterations = env_count("GSTSMITH_BENCH_ITERATIONS", 100);
    let caps = caps();
    let info = gst_inference_common::model_info::ModelInfo::parse(MODEL_INFO)
        .map_err(std::io::Error::other)?;
    let video_info = gst_video::VideoInfo::from_caps(&caps)?;
    let source = input_buffer(&caps).map_readable()?.as_slice().to_vec();
    for factory in ["tractinference", "ortinference"] {
        let (element, directory) = fixture_element(factory)?;
        let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
        harness.set_src_caps(caps.clone());
        harness.play();
        for _ in 0..warmup {
            harness.push_and_pull(input_buffer(&caps))?;
        }
        let preprocessing_start = Instant::now();
        for _ in 0..iterations {
            let _ = gst_inference_common::preprocess::preprocess(
                &source,
                usize::try_from(video_info.stride()[0])?,
                2,
                1,
                gst_inference_common::preprocess::PixelFormat::Rgb,
                gst_inference_common::preprocess::ChannelOrder::Rgb,
                info.input(),
            )?;
        }
        let sample_count = f64::from(u32::try_from(iterations)?);
        let preprocessing = preprocessing_start.elapsed().div_f64(sample_count);
        let total_start = Instant::now();
        for _ in 0..iterations {
            harness.push_and_pull(input_buffer(&caps))?;
        }
        let total = total_start.elapsed().div_f64(sample_count);
        let inference = total.saturating_sub(preprocessing);
        println!(
            "{factory}: warmup={warmup} iterations={iterations} preprocessing={preprocessing:?} inference={inference:?} total={total:?}"
        );
        drop(directory);
    }
    Ok(())
}
