#![expect(
    clippy::expect_used,
    reason = "GStreamer harness setup failures should fail the integration test"
)]

use std::fs;
use std::sync::Once;

use gst::prelude::*;

const MODEL_INFO: &str =
    include_str!("../../inference-common/tests/fixtures/masked-sequence.onnx.modelinfo");
const MODEL: &[u8] = include_bytes!("../../inference-common/tests/fixtures/masked-sequence.onnx");
const FACTORIES: [&str; 2] = ["ortinference", "tractinference"];

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstortinference::plugin_register_static().expect("registering ORT inference plugin");
        gsttractinference::plugin_register_static().expect("registering Tract inference plugin");
    });
}

fn harness(factory: &str) -> (gst_check::Harness, tempfile::TempDir) {
    fixture_harness(factory, MODEL, MODEL_INFO, "tensor-meta")
}

fn fixture_harness(
    factory: &str,
    model_bytes: &[u8],
    model_info: &str,
    mode: &str,
) -> (gst_check::Harness, tempfile::TempDir) {
    init();
    let directory = tempfile::tempdir().expect("creating fixture directory");
    let model = directory.path().join("fixture.onnx");
    fs::write(&model, model_bytes).expect("writing fixture model");
    fs::write(directory.path().join("fixture.onnx.modelinfo"), model_info)
        .expect("writing fixture model-info");
    let element = gst::ElementFactory::make(factory)
        .property("model-file", model.to_string_lossy().as_ref())
        .property_from_str("input-mode", mode)
        .build()
        .expect("creating tensor inference element");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    if mode == "video" {
        harness.set_src_caps_str("video/x-raw,format=RGB,width=2,height=1,framerate=1/1");
    } else {
        harness.set_src_caps(gst::Caps::new_empty_simple("application/x-test"));
    }
    harness.play();
    (harness, directory)
}

fn assert_axis_contract(
    h: &gst_check::Harness,
    output: &gst::BufferRef,
    contract: gst_inference_common::model_info::CapsContract<'_>,
) {
    let caps = h.sinkpad().expect("sink pad").current_caps().expect("caps");
    let groups = caps
        .structure(0)
        .expect("structure")
        .get::<gst::Structure>("tensors")
        .expect("tensor groups");
    let descriptors = groups
        .get::<gst::UniqueList>(contract.group_id)
        .expect("descriptors");
    assert_eq!(descriptors.len(), contract.outputs.len());
    for (description, caps) in contract.outputs.iter().zip(descriptors.iter()) {
        let caps = caps.get::<gst::Caps>().expect("tensor caps");
        let structure = caps.structure(0).expect("tensor structure");
        assert_eq!(
            structure.get::<String>("tensor-id").expect("id"),
            description.id
        );
        assert_eq!(structure.get::<String>("type").expect("type"), "float32");
        assert_eq!(
            structure.get::<String>("dims-order").expect("order"),
            "row-major"
        );
        let dims = structure.get::<gst::Array>("dims").expect("dims");
        let dims = dims
            .iter()
            .map(|v| {
                usize::try_from(v.get::<i32>().expect("dimension")).expect("positive dimension")
            })
            .collect::<Vec<_>>();
        assert_eq!(dims, description.dims);
        let tensor = output
            .iter_meta::<gst_analytics::TensorMeta>()
            .flat_map(|meta| meta.as_slice().to_vec())
            .find(|tensor| tensor.id().as_str() == description.id.as_str())
            .expect("output tensor");
        assert_eq!(tensor.dims(), description.dims);
        assert_eq!(tensor.data_type(), gst_analytics::TensorDataType::Float32);
        assert_eq!(tensor.dims_order(), gst_analytics::TensorDimOrder::RowMajor);
        let expected = (1..=description.dims.iter().product::<usize>())
            .map(|v| f32::from(u16::try_from(v).expect("small fixture")))
            .collect::<Vec<_>>();
        assert_eq!(float_values(output, &description.id), expected);
    }
}

#[test]
fn non_unit_leading_axes_survive_tensor_inference() {
    use gst_inference_common::model_info::TensorModelInfo;
    let contents = include_str!("../../inference-common/tests/fixtures/tensor-axes.onnx.modelinfo");
    let info = TensorModelInfo::parse(contents).expect("arbitrary axes parse");
    for factory in FACTORIES {
        let (mut h, _directory) = fixture_harness(
            factory,
            include_bytes!("../../inference-common/tests/fixtures/tensor-axes.onnx"),
            contents,
            "tensor-meta",
        );
        let inputs = info
            .inputs()
            .iter()
            .rev()
            .map(|input| {
                let bytes = (1..=input.dims.iter().product::<usize>())
                    .flat_map(|v| f32::from(u16::try_from(v).expect("small fixture")).to_le_bytes())
                    .collect();
                tensor(
                    &input.id,
                    gst_analytics::TensorDataType::Float32,
                    &input.dims,
                    bytes,
                )
            })
            .collect();
        let output = h.push_and_pull(buffer(inputs)).expect("tensor inference");
        assert_axis_contract(&h, &output, info.caps_contract());
    }
}

#[test]
fn video_outputs_have_arbitrary_axes_and_truthful_memory_order() {
    use gst_inference_common::model_info::ModelInfo;
    let contents =
        include_str!("../../inference-common/tests/fixtures/image-reshape.onnx.modelinfo");
    let info = ModelInfo::parse(contents).expect("unbatched outputs parse");
    for factory in FACTORIES {
        let (mut h, _directory) = fixture_harness(
            factory,
            include_bytes!("../../inference-common/tests/fixtures/image-reshape.onnx"),
            contents,
            "video",
        );
        let input = gst::Buffer::from_mut_slice(vec![1_u8, 2, 3, 4, 5, 6, 0, 0]);
        let output = h.push_and_pull(input).expect("video inference");
        assert_axis_contract(&h, &output, info.caps_contract());
    }
}

fn tensor(
    id: &str,
    data_type: gst_analytics::TensorDataType,
    dims: &[usize],
    bytes: Vec<u8>,
) -> gst_analytics::Tensor {
    gst_analytics::Tensor::new_simple(
        gst::glib::Quark::from_str(id),
        data_type,
        gst::Buffer::from_mut_slice(bytes),
        gst_analytics::TensorDimOrder::RowMajor,
        dims,
    )
}

fn sequence(dims: &[usize]) -> gst_analytics::Tensor {
    let count = dims.iter().product::<usize>();
    let values = (1..=count)
        .flat_map(|value| f32::from(u16::try_from(value).expect("small fixture")).to_le_bytes())
        .collect();
    tensor(
        "sequence",
        gst_analytics::TensorDataType::Float32,
        dims,
        values,
    )
}

fn mask() -> gst_analytics::Tensor {
    tensor(
        "sequence-mask",
        gst_analytics::TensorDataType::Uint8,
        &[1, 3],
        vec![1, 0, 1],
    )
}

fn scale(value: f32) -> gst_analytics::Tensor {
    tensor(
        "scale",
        gst_analytics::TensorDataType::Float32,
        &[1],
        value.to_le_bytes().to_vec(),
    )
}

fn buffer(tensors: Vec<gst_analytics::Tensor>) -> gst::Buffer {
    let mut buffer = gst::Buffer::from_slice(vec![7_u8; 4]);
    if !tensors.is_empty() {
        gst_analytics::TensorMeta::add(buffer.get_mut().expect("writable buffer"))
            .set(tensors.into());
    }
    buffer
}

fn tensor_ids(buffer: &gst::BufferRef) -> Vec<String> {
    buffer
        .iter_meta::<gst_analytics::TensorMeta>()
        .flat_map(|meta| meta.as_slice().to_vec())
        .map(|tensor| tensor.id().as_str().to_string())
        .collect()
}

fn float_values(buffer: &gst::BufferRef, id: &str) -> Vec<f32> {
    let id = gst::glib::Quark::from_str(id);
    let tensor = buffer
        .iter_meta::<gst_analytics::TensorMeta>()
        .flat_map(|meta| meta.as_slice().to_vec())
        .find(|tensor| tensor.id() == id)
        .expect("output tensor attached");
    let map = tensor.data().map_readable().expect("mapping output tensor");
    map.as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[test]
fn fixture_metadata_matches_native_gstreamer_model_info() {
    use gst_analytics::ModelInfoTensorDirection::{Input, Output};
    use gst_inference_common::{model_info::TensorModelInfo, tensor::tensor_data_type};

    init();
    let directory = tempfile::tempdir().expect("creating fixture directory");
    let model = directory.path().join("sequence.onnx");
    fs::write(directory.path().join("sequence.onnx.modelinfo"), MODEL_INFO)
        .expect("writing fixture model-info");
    let native = gst_analytics::ModelInfo::load(&model).expect("native model-info load");
    let info = TensorModelInfo::parse(MODEL_INFO).expect("shared model-info parse");
    assert_eq!(native.version().as_str(), "1.0");
    assert_eq!(native.group_id().as_deref(), Some(info.group_id()));
    for (direction, tensors) in [(Input, info.inputs()), (Output, info.outputs())] {
        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(native.id(&tensor.name).as_deref(), Some(tensor.id.as_str()));
            assert_eq!(
                native
                    .find_tensor_name(
                        direction,
                        index,
                        Some(&tensor.name),
                        tensor_data_type(tensor.data_type),
                        &tensor.dims,
                    )
                    .as_deref(),
                Some(tensor.name.as_str())
            );
        }
    }
}

#[test]
fn runs_on_upstream_tensors_and_keeps_them() {
    for factory in FACTORIES {
        let (mut h, _directory) = harness(factory);
        // Selection follows ids across metadata, not tensor order or position.
        let mut input = buffer(vec![mask(), scale(2.0)]);
        gst_analytics::TensorMeta::add(input.get_mut().expect("writable buffer"))
            .set(vec![sequence(&[1, 3, 2])].into());
        let output = h.push_and_pull(input).expect("running tensor model");
        assert_eq!(
            output.map_readable().expect("readable").as_slice(),
            &[7; 4],
            "{factory}: payload passes through"
        );
        assert_eq!(
            tensor_ids(&output),
            ["sequence-mask", "scale", "sequence", "scaled"],
            "{factory}: inputs stay, only the declared output is added"
        );
        assert_eq!(
            float_values(&output, "scaled"),
            [2.0, 4.0, 0.0, 0.0, 10.0, 12.0],
            "{factory}: y = x * mask * scale(upstream 2)"
        );
        let caps = h
            .sinkpad()
            .expect("harness sink pad")
            .current_caps()
            .expect("caps negotiated");
        let groups = caps
            .structure(0)
            .expect("caps structure")
            .get::<gst::Structure>("tensors")
            .expect("tensor groups advertised");
        assert!(
            groups.has_field("gstsmith-masked-sequence-fixture"),
            "{factory}: {caps}"
        );
        let output = h
            .push_and_pull(buffer(vec![sequence(&[1, 3, 2]), mask(), scale(3.0)]))
            .expect("running with a different upstream scale");
        assert_eq!(
            float_values(&output, "scaled"),
            [3.0, 6.0, 0.0, 0.0, 15.0, 18.0]
        );
    }
}

#[test]
fn buffers_without_inputs_pass_through_untouched() {
    for factory in FACTORIES {
        let (mut h, _directory) = harness(factory);
        let output = h
            .push_and_pull(buffer(Vec::new()))
            .expect("passing through");
        assert!(tensor_ids(&output).is_empty(), "{factory}");
        let unrelated = h
            .push_and_pull(buffer(vec![tensor(
                "other",
                gst_analytics::TensorDataType::Uint8,
                &[1],
                vec![1],
            )]))
            .expect("passing through");
        assert_eq!(tensor_ids(&unrelated), ["other"], "{factory}");
    }
}

#[test]
fn partial_or_mismatched_inputs_are_errors() {
    init();
    for factory in FACTORIES {
        for tensors in [
            vec![sequence(&[1, 3, 2])],
            vec![sequence(&[1, 3, 2]), mask()],
            vec![sequence(&[1, 2, 2]), mask(), scale(2.0)],
            vec![
                tensor(
                    "sequence",
                    gst_analytics::TensorDataType::Int32,
                    &[1, 3, 2],
                    vec![0; 24],
                ),
                mask(),
                scale(2.0),
            ],
        ] {
            let (mut h, _directory) = harness(factory);
            assert_eq!(
                h.push(buffer(tensors)),
                Err(gst::FlowError::Error),
                "{factory}"
            );
        }
    }
}

#[test]
fn duplicate_required_ids_are_errors_within_and_across_metadata() {
    for factory in FACTORIES {
        for separate_meta in [false, true] {
            let (mut h, _directory) = harness(factory);
            let mut input = if separate_meta {
                buffer(vec![sequence(&[1, 3, 2]), mask(), scale(2.0)])
            } else {
                buffer(vec![
                    sequence(&[1, 3, 2]),
                    mask(),
                    scale(2.0),
                    sequence(&[1, 3, 2]),
                ])
            };
            if separate_meta {
                gst_analytics::TensorMeta::add(input.get_mut().expect("writable buffer"))
                    .set(vec![sequence(&[1, 3, 2])].into());
            }
            assert_eq!(h.push(input), Err(gst::FlowError::Error), "{factory}");
        }
    }
}

#[test]
fn input_mode_controls_caps_before_startup() {
    init();
    let video = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .build();
    let unsupported_video = gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .build();
    let other = gst::Caps::new_empty_simple("application/x-test");
    for factory in FACTORIES {
        let element = gst::ElementFactory::make(factory)
            .build()
            .expect("creating inference element");
        let property = element
            .find_property("input-mode")
            .expect("input mode property");
        assert!(property.flags().contains(gst::PARAM_FLAG_MUTABLE_READY));
        let value = element.property_value("input-mode");
        let (_, mode) = gst::glib::EnumValue::from_value(&value).expect("input mode enum");
        assert_eq!(mode.nick(), "video");
        for pad_name in ["sink", "src"] {
            let pad = element.static_pad(pad_name).expect("inference pad");
            let caps = pad.query_caps(None);
            assert!(caps.can_intersect(&video), "{factory}: {caps}");
            assert!(!caps.can_intersect(&unsupported_video), "{factory}: {caps}");
            assert!(!caps.can_intersect(&other), "{factory}: {caps}");
            element.set_property_from_str("input-mode", "tensor-meta");
            assert!(pad.query_caps(None).can_intersect(&other), "{factory}");
            element.set_property_from_str("input-mode", "video");
            assert!(!pad.query_caps(None).can_intersect(&other), "{factory}");
        }
    }
}

#[test]
fn secondary_inference_selects_primary_output_and_preserves_the_carrier() {
    init();
    let directory = tempfile::tempdir().expect("creating fixture directory");
    let model = directory.path().join("identity.onnx");
    let primary_info = directory.path().join("primary.modelinfo");
    let secondary_info = directory.path().join("secondary.modelinfo");
    let info = include_str!("../../inference-common/tests/fixtures/identity.onnx.modelinfo");
    fs::write(
        &model,
        include_bytes!("../../inference-common/tests/fixtures/identity.onnx"),
    )
    .expect("writing model");
    fs::write(&primary_info, info).expect("writing primary model-info");
    fs::write(
        &secondary_info,
        info.replace("group-id=gstsmith-identity-fixture", "group-id=secondary")
            .replace("id=first\n", "id=secondary-first\n")
            .replace("id=second\n", "id=secondary-second\n")
            .replace("id=image", "id=second")
            .replace("ranges=0.0,255.0\n", ""),
    )
    .expect("writing secondary model-info");
    for primary in FACTORIES {
        for secondary in FACTORIES {
            let mut h = gst_check::Harness::new_parse(&format!(
                "{primary} model-file=\"{}\" model-info-file=\"{}\" ! \
                 {secondary} input-mode=tensor-meta model-file=\"{}\" model-info-file=\"{}\"",
                model.display(),
                primary_info.display(),
                model.display(),
                secondary_info.display()
            ));
            h.set_src_caps_str("video/x-raw,format=RGB,width=2,height=1,framerate=1/1");
            h.play();
            let mut input = gst::Buffer::from_mut_slice(vec![1_u8, 2, 3, 4, 5, 6, 0, 0]);
            let input_ref = input.get_mut().expect("writable input");
            input_ref.set_pts(gst::ClockTime::SECOND);
            input_ref.set_duration(gst::ClockTime::SECOND);
            input_ref.set_flags(gst::BufferFlags::DISCONT);
            // A video primary must still use pixels even when matching tensor
            // metadata already exists. Secondary selection must ignore it.
            gst_analytics::TensorMeta::add(input_ref).set(
                vec![tensor(
                    "image",
                    gst_analytics::TensorDataType::Float32,
                    &[1, 1, 2, 3],
                    vec![0; 24],
                )]
                .into(),
            );
            let output = h
                .push_and_pull(input)
                .expect("running primary/secondary chain");
            assert_eq!(
                output.map_readable().expect("readable output").as_slice(),
                &[1, 2, 3, 4, 5, 6, 0, 0]
            );
            assert_eq!(output.pts(), Some(gst::ClockTime::SECOND));
            assert_eq!(output.duration(), Some(gst::ClockTime::SECOND));
            assert!(output.flags().contains(gst::BufferFlags::DISCONT));
            assert_eq!(
                tensor_ids(&output),
                [
                    "image",
                    "first",
                    "second",
                    "secondary-first",
                    "secondary-second"
                ]
            );
            assert_eq!(
                float_values(&output, "secondary-first"),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
            let caps = h
                .sinkpad()
                .expect("harness sink pad")
                .current_caps()
                .expect("negotiated caps");
            let groups = caps
                .structure(0)
                .expect("caps structure")
                .get::<gst::Structure>("tensors")
                .expect("tensor groups");
            assert!(
                groups.has_field("gstsmith-identity-fixture"),
                "{primary} -> {secondary}: {caps}"
            );
            assert!(
                groups.has_field("secondary"),
                "{primary} -> {secondary}: {caps}"
            );
        }
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_tensor_inference_uses_mlprogram_options() {
    init();
    let directory = tempfile::tempdir().expect("creating fixture directory");
    let model = directory.path().join("conv.onnx");
    fs::write(
        &model,
        include_bytes!("../../tract-inference/tests/fixtures/metal-conv.onnx"),
    )
    .expect("writing convolution model");
    fs::write(
        directory.path().join("conv.onnx.modelinfo"),
        include_str!("../../tract-inference/tests/fixtures/metal-conv.onnx.modelinfo")
            .replace("ranges=0.0,255.0\n", ""),
    )
    .expect("writing tensor model-info");
    let cache = directory.path().join("coreml-cache");
    let element = gst::ElementFactory::make("ortinference")
        .property("model-file", model.to_string_lossy().as_ref())
        .property_from_str("input-mode", "tensor-meta")
        .property_from_str("execution-provider", "coreml")
        .property("strict-execution-provider", true)
        .property_from_str("coreml-model-format", "mlprogram")
        .property_from_str("coreml-compute-units", "cpu-only")
        .property(
            "coreml-model-cache-directory",
            cache.to_string_lossy().as_ref(),
        )
        .build()
        .expect("creating tensor inference element");
    let mut h = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    h.set_src_caps(gst::Caps::new_empty_simple("application/x-test"));
    h.play();
    let output = h
        .push_and_pull(buffer(vec![tensor(
            "image",
            gst_analytics::TensorDataType::Float32,
            &[1, 3, 2, 2],
            [1.0_f32; 12]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )]))
        .expect("running CoreML tensor inference");
    assert_eq!(float_values(&output, "convolution"), [6.5; 4]);
    assert!(cache.is_dir(), "CoreML cache option must reach the session");
}
