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
        assert_eq!(tensor_values(tensor), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    let negotiated = element
        .static_pad("src")
        .ok_or_else(|| std::io::Error::other("source pad is missing"))?
        .current_caps()
        .ok_or_else(|| std::io::Error::other("source caps were not negotiated"))?;
    let structure = negotiated
        .structure(0)
        .ok_or_else(|| std::io::Error::other("source caps structure is missing"))?;
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
    let (element, directory) = fixture_element(factory)?;
    let caps = caps();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(caps.clone());
    harness.play();
    let output = harness.push_and_pull(input_buffer(&caps))?;
    assert_tensor_contract(&element, &output)?;
    Ok((output, caps, directory))
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

#[test]
fn properties_have_backend_defaults_and_ready_mutability() -> Result<(), Box<dyn std::error::Error>>
{
    init();
    let element = gst::ElementFactory::make("ortinference").build()?;
    assert_eq!(enum_nick(&element, "execution-provider"), "cpu");
    assert_eq!(element.property::<u32>("intra-op-threads"), 0);
    assert_eq!(enum_nick(&element, "graph-optimization"), "level3");
    for name in [
        "model-file",
        "model-info-file",
        "execution-provider",
        "intra-op-threads",
        "graph-optimization",
    ] {
        let property = element
            .find_property(name)
            .ok_or_else(|| std::io::Error::other("missing property"))?;
        assert!(
            property.flags().contains(gst::PARAM_FLAG_MUTABLE_READY),
            "{name} must be READY-mutable"
        );
    }
    Ok(())
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
