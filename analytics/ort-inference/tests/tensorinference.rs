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
const FACTORIES: [&str; 2] = ["orttensorinference", "tracttensorinference"];

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstortinference::plugin_register_static().expect("registering ORT inference plugin");
        gsttractinference::plugin_register_static().expect("registering Tract inference plugin");
    });
}

fn harness(factory: &str) -> (gst_check::Harness, tempfile::TempDir) {
    init();
    let directory = tempfile::tempdir().expect("creating fixture directory");
    let model = directory.path().join("masked-sequence.onnx");
    fs::write(&model, MODEL).expect("writing fixture model");
    fs::write(
        directory.path().join("masked-sequence.onnx.modelinfo"),
        MODEL_INFO,
    )
    .expect("writing fixture model-info");
    let element = gst::ElementFactory::make(factory)
        .property("model-file", model.to_string_lossy().as_ref())
        .build()
        .expect("creating tensor inference element");
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(gst::Caps::new_empty_simple("application/x-test"));
    harness.play();
    (harness, directory)
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
    // `bool` tensors travel as `uint8` 0/1 (GStreamer 1.28 cannot build bool
    // tensors).
    tensor(
        "sequence-mask",
        gst_analytics::TensorDataType::Uint8,
        &[1, 3],
        vec![1, 0, 1],
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
fn runs_on_upstream_tensors_and_keeps_them() {
    for factory in FACTORIES {
        let (mut h, _directory) = harness(factory);
        let output = h
            .push_and_pull(buffer(vec![sequence(&[1, 3, 2]), mask()]))
            .expect("running tensor model");
        assert_eq!(
            output.map_readable().expect("readable").as_slice(),
            &[7; 4],
            "{factory}: payload passes through"
        );
        assert_eq!(
            tensor_ids(&output),
            ["sequence", "sequence-mask", "scaled"],
            "{factory}: inputs stay, only the declared output is added"
        );
        assert_eq!(
            float_values(&output, "scaled"),
            [2.0, 4.0, 0.0, 0.0, 10.0, 12.0],
            "{factory}: y = x * mask * scale(constant 2)"
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
            vec![sequence(&[1, 2, 2]), mask()],
            vec![
                tensor(
                    "sequence",
                    gst_analytics::TensorDataType::Int32,
                    &[1, 3, 2],
                    vec![0; 24],
                ),
                mask(),
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
