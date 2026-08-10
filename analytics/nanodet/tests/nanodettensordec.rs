#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test setup uses fixed synthetic tensor dimensions and fatal assertions"
)]

use std::fs;
use std::sync::Once;

use gst::prelude::*;
use gst_analytics::AnalyticsMetaRefExt;

const TENSOR_ID: &str = "nanodet-output";
const GROUP_ID: &str = "gstsmith-nanodet";
const CLASSES: usize = 80;
const BINS: usize = 8;
const CHANNELS: usize = CLASSES + 4 * BINS;

#[derive(Clone, Copy)]
struct Contract {
    input_size: usize,
    points: usize,
}

const CONTRACTS: [Contract; 4] = [
    Contract {
        input_size: 320,
        points: 2_100,
    },
    Contract {
        input_size: 320,
        points: 2_125,
    },
    Contract {
        input_size: 416,
        points: 3_549,
    },
    Contract {
        input_size: 416,
        points: 3_598,
    },
];

#[derive(Clone, Copy)]
enum TensorKind {
    Float32,
    Float16,
}

impl TensorKind {
    const fn caps_name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
        }
    }

    const fn data_type(self) -> gst_analytics::TensorDataType {
        match self {
            Self::Float32 => gst_analytics::TensorDataType::Float32,
            Self::Float16 => gst_analytics::TensorDataType::Float16,
        }
    }
}

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstnanodet::plugin_register_static().expect("registering NanoDet plugin");
    });
}

fn tensor_caps(contract: Contract, kind: TensorKind, tensor_id: &str) -> gst::Caps {
    gst::Caps::builder("tensor/strided")
        .field(
            "dims",
            gst::Array::from_values([
                1i32.to_send_value(),
                i32::try_from(contract.points)
                    .expect("points fit i32")
                    .to_send_value(),
                i32::try_from(CHANNELS)
                    .expect("channels fit i32")
                    .to_send_value(),
            ]),
        )
        .field("dims-order", "row-major")
        .field("type", kind.caps_name())
        .field("tensor-id", tensor_id)
        .build()
}

fn video_caps(contract: Contract, kind: TensorKind, tensor_id: &str) -> gst::Caps {
    let mut groups = gst::Structure::new_empty("tensorgroups");
    groups.set(
        GROUP_ID,
        gst::UniqueList::new([tensor_caps(contract, kind, tensor_id)]),
    );
    let input_size = i32::try_from(contract.input_size).expect("input size fits i32");
    gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", input_size)
        .field("height", input_size)
        .field("framerate", gst::Fraction::new(1, 1))
        .field("tensors", groups)
        .build()
}

fn empty_video_buffer(contract: Contract) -> gst::Buffer {
    gst::Buffer::with_size(contract.input_size * contract.input_size * 3)
        .expect("allocate video frame")
}

fn synthetic_values(contract: Contract) -> Vec<f32> {
    vec![0.0; contract.points * CHANNELS]
}

fn add_candidate(values: &mut [f32], point: usize, class: usize, score: f32) {
    let row = point * CHANNELS;
    values[row + class] = score;
    for side in 0..4 {
        for bin in 0..BINS {
            values[row + CLASSES + side * BINS + bin] = if bin == 1 { 20.0 } else { 0.0 };
        }
    }
}

fn tensor_bytes(values: &[f32], kind: TensorKind) -> Vec<u8> {
    match kind {
        TensorKind::Float32 => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        TensorKind::Float16 => values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_ne_bytes())
            .collect(),
    }
}

fn attach_tensor(
    buffer: &mut gst::BufferRef,
    id: &str,
    data_type: gst_analytics::TensorDataType,
    order: gst_analytics::TensorDimOrder,
    dims: &[usize],
    bytes: Vec<u8>,
) {
    let tensor = gst_analytics::Tensor::new_simple(
        gst::glib::Quark::from_str(id),
        data_type,
        gst::Buffer::from_mut_slice(bytes),
        order,
        dims,
    );
    let mut meta = gst_analytics::TensorMeta::add(buffer);
    meta.set([tensor].into());
}

fn harness(element: &gst::Element, contract: Contract, kind: TensorKind) -> gst_check::Harness {
    let mut harness = gst_check::Harness::with_element(element, Some("sink"), Some("src"));
    harness.set_src_caps(video_caps(contract, kind, TENSOR_ID));
    harness.play();
    harness
}

#[test]
fn registers_factory_and_exposes_all_contracts() {
    init();
    let element = gst::ElementFactory::make("nanodettensordec")
        .build()
        .expect("construct decoder");
    assert_eq!(element.property::<String>("tensor-id"), TENSOR_ID);
    assert!((element.property::<f32>("score-threshold") - 0.3).abs() < f32::EPSILON);
    assert!((element.property::<f32>("iou-threshold") - 0.6).abs() < f32::EPSILON);
    assert_eq!(element.property::<u32>("max-detections"), 100);

    let caps = element
        .static_pad("sink")
        .expect("sink pad")
        .pad_template_caps();
    assert_eq!(caps.size(), CONTRACTS.len() * 2);
    for contract in CONTRACTS {
        for kind in [TensorKind::Float32, TensorKind::Float16] {
            assert!(caps.can_intersect(&video_caps(contract, kind, TENSOR_ID)));
        }
    }
}

#[test]
fn missing_metadata_and_a_different_tensor_id_pass_through() {
    init();
    let contract = CONTRACTS[1];
    let element = gst::ElementFactory::make("nanodettensordec")
        .build()
        .expect("construct decoder");
    let mut harness = harness(&element, contract, TensorKind::Float32);
    let output = harness
        .push_and_pull(empty_video_buffer(contract))
        .expect("missing metadata passes through");
    assert!(
        output
            .meta::<gst_analytics::AnalyticsRelationMeta>()
            .is_none()
    );

    let mut input = empty_video_buffer(contract);
    attach_tensor(
        input.get_mut().expect("writable buffer"),
        "another-output",
        gst_analytics::TensorDataType::Float32,
        gst_analytics::TensorDimOrder::RowMajor,
        &[1, contract.points, CHANNELS],
        tensor_bytes(&synthetic_values(contract), TensorKind::Float32),
    );
    let output = harness
        .push_and_pull(input)
        .expect("unconfigured tensor passes through");
    assert!(
        output
            .meta::<gst_analytics::AnalyticsRelationMeta>()
            .is_none()
    );
}

#[test]
fn decodes_every_model_contract_and_float_type() {
    init();
    for contract in CONTRACTS {
        for kind in [TensorKind::Float32, TensorKind::Float16] {
            let element = gst::ElementFactory::make("nanodettensordec")
                .build()
                .expect("construct decoder");
            let mut harness = harness(&element, contract, kind);
            let mut values = synthetic_values(contract);
            add_candidate(&mut values, 0, 7, 0.9);
            let mut input = empty_video_buffer(contract);
            attach_tensor(
                input.get_mut().expect("writable buffer"),
                TENSOR_ID,
                kind.data_type(),
                gst_analytics::TensorDimOrder::RowMajor,
                &[1, contract.points, CHANNELS],
                tensor_bytes(&values, kind),
            );
            let output = harness
                .push_and_pull(input)
                .expect("decode synthetic tensor");
            let relation = output
                .meta::<gst_analytics::AnalyticsRelationMeta>()
                .expect("analytics relation metadata");
            assert_eq!(relation.len(), 2);
            let object = relation
                .iter::<gst_analytics::AnalyticsODMtd>()
                .next()
                .expect("one object detection");
            assert_eq!(object.obj_type().expect("object type").as_str(), "class-7");
            assert!(object.location().expect("object location").loc_conf_lvl > 0.89);
            assert_eq!(
                relation
                    .iter_direct_related::<gst_analytics::AnalyticsClassificationMtd>(
                        object.id(),
                        gst_analytics::RelTypes::RELATE_TO,
                    )
                    .count(),
                1
            );
        }
    }
}

#[test]
fn raw_int8_tensor_fails_streaming() {
    init();
    let contract = CONTRACTS[0];
    let element = gst::ElementFactory::make("nanodettensordec")
        .build()
        .expect("construct decoder");
    let mut harness = harness(&element, contract, TensorKind::Float32);
    let mut input = empty_video_buffer(contract);
    attach_tensor(
        input.get_mut().expect("writable buffer"),
        TENSOR_ID,
        gst_analytics::TensorDataType::Int8,
        gst_analytics::TensorDimOrder::RowMajor,
        &[1, contract.points, CHANNELS],
        vec![0; contract.points * CHANNELS],
    );
    assert_eq!(harness.push(input), Err(gst::FlowError::Error));
}

#[test]
fn accepts_an_eighty_label_file_and_rejects_bad_files() {
    init();
    let contract = CONTRACTS[1];
    let directory = tempfile::tempdir().expect("temporary directory");
    let valid = directory.path().join("labels.txt");
    let labels = (0..CLASSES)
        .map(|index| format!("label-{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&valid, labels).expect("write valid labels");
    let element = gst::ElementFactory::make("nanodettensordec")
        .property("label-file", valid.to_string_lossy().as_ref())
        .build()
        .expect("construct decoder");
    let mut harness = harness(&element, contract, TensorKind::Float32);
    let mut values = synthetic_values(contract);
    add_candidate(&mut values, 0, 7, 0.9);
    let mut input = empty_video_buffer(contract);
    attach_tensor(
        input.get_mut().expect("writable buffer"),
        TENSOR_ID,
        gst_analytics::TensorDataType::Float32,
        gst_analytics::TensorDimOrder::RowMajor,
        &[1, contract.points, CHANNELS],
        tensor_bytes(&values, TensorKind::Float32),
    );
    let output = harness.push_and_pull(input).expect("decode with labels");
    let relation = output
        .meta::<gst_analytics::AnalyticsRelationMeta>()
        .expect("analytics metadata");
    assert_eq!(
        relation
            .iter::<gst_analytics::AnalyticsODMtd>()
            .next()
            .expect("object")
            .obj_type()
            .expect("label")
            .as_str(),
        "label-7"
    );

    let missing = directory.path().join("missing.txt");
    let missing_element = gst::ElementFactory::make("nanodettensordec")
        .property("label-file", missing.to_string_lossy().as_ref())
        .build()
        .expect("construct decoder");
    let missing_pipeline = gst::Pipeline::new();
    missing_pipeline.add(&missing_element).expect("add decoder");
    assert_eq!(
        missing_pipeline.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    missing_pipeline
        .set_state(gst::State::Null)
        .expect("stop missing-label pipeline");

    let wrong = directory.path().join("wrong.txt");
    fs::write(&wrong, "one\ntwo\n").expect("write wrong labels");
    let wrong_element = gst::ElementFactory::make("nanodettensordec")
        .property("label-file", wrong.to_string_lossy().as_ref())
        .build()
        .expect("construct decoder");
    let wrong_pipeline = gst::Pipeline::new();
    wrong_pipeline.add(&wrong_element).expect("add decoder");
    assert_eq!(
        wrong_pipeline.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    );
    wrong_pipeline
        .set_state(gst::State::Null)
        .expect("stop wrong-label pipeline");
}
