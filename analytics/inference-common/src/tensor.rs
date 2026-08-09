use gst::glib;
use gst::prelude::*;

use crate::engine::OwnedTensor;
use crate::model_info::{DimOrder, ModelInfo, ScalarType, TensorDescription};

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "model-info parsing rejects dimensions outside the i32 GStreamer caps range"
)]
#[must_use]
pub fn tensor_caps(tensor: &TensorDescription) -> gst::Caps {
    gst::Caps::builder("tensor/strided")
        .field(
            "dims",
            gst::Array::from_values(
                tensor
                    .dims
                    .iter()
                    .map(|dimension| (*dimension as i32).to_send_value()),
            ),
        )
        .field("dims-order", tensor.dim_order.as_caps_name())
        .field("type", tensor.data_type.as_caps_name())
        .field("tensor-id", tensor.id.as_str())
        .build()
}

pub fn attach_tensors(buffer: &mut gst::BufferRef, outputs: Vec<OwnedTensor>) {
    let tensors = outputs
        .into_iter()
        .map(|output| {
            let data = gst::Buffer::from_mut_slice(output.bytes);
            let data_type = tensor_data_type(output.description.data_type);
            let order = match output.description.dim_order {
                DimOrder::RowMajor => gst_analytics::TensorDimOrder::RowMajor,
                DimOrder::ColMajor => gst_analytics::TensorDimOrder::ColMajor,
            };
            gst_analytics::Tensor::new_simple(
                glib::Quark::from_str(&output.description.id),
                data_type,
                data,
                order,
                &output.description.dims,
            )
        })
        .collect::<Vec<_>>();
    let mut meta = gst_analytics::TensorMeta::add(buffer);
    meta.set(tensors.into());
}

#[must_use]
pub fn tensor_data_type(data_type: ScalarType) -> gst_analytics::TensorDataType {
    match data_type {
        ScalarType::Float16 => gst_analytics::TensorDataType::Float16,
        ScalarType::Float64 => gst_analytics::TensorDataType::Float64,
        ScalarType::Float32 => gst_analytics::TensorDataType::Float32,
        ScalarType::Int8 => gst_analytics::TensorDataType::Int8,
        ScalarType::Int16 => gst_analytics::TensorDataType::Int16,
        ScalarType::Int32 => gst_analytics::TensorDataType::Int32,
        ScalarType::Int64 => gst_analytics::TensorDataType::Int64,
        ScalarType::Uint8 => gst_analytics::TensorDataType::Uint8,
        ScalarType::Uint16 => gst_analytics::TensorDataType::Uint16,
        ScalarType::Uint32 => gst_analytics::TensorDataType::Uint32,
        ScalarType::Uint64 => gst_analytics::TensorDataType::Uint64,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "model-info parsing rejects dimensions outside the i32 GStreamer caps range"
)]
#[must_use]
pub fn transform_caps(
    info: Option<&ModelInfo>,
    direction: gst::PadDirection,
    caps: &gst::Caps,
    filter: Option<&gst::Caps>,
) -> gst::Caps {
    let mut result = caps.copy();
    for structure in result.make_mut().iter_mut() {
        if let Some((width, height)) = info.and_then(|info| info.image_dimensions().ok()) {
            structure.set("width", width as i32);
            structure.set("height", height as i32);
        }
        if direction == gst::PadDirection::Src {
            structure.remove_field("tensors");
        } else if let Some(info) = info {
            let mut groups = structure
                .get::<gst::Structure>("tensors")
                .unwrap_or_else(|_| gst::Structure::new_empty("tensorgroups"));
            // Sink caps cannot already contain this element's output group. If
            // they do, the tensor groups came from downstream query feedback.
            if groups.has_field(info.group_id()) {
                groups = gst::Structure::new_empty("tensorgroups");
            }
            let descriptors = info.outputs().iter().map(tensor_caps).collect::<Vec<_>>();
            groups.set(info.group_id(), gst::UniqueList::new(descriptors));
            structure.set("tensors", groups);
        }
    }
    filter.map_or(result.clone(), |filter| {
        filter.intersect_with_mode(&result, gst::CapsIntersectMode::First)
    })
}

#[must_use]
pub fn fixate_caps(
    info: Option<&ModelInfo>,
    direction: gst::PadDirection,
    caps: &gst::Caps,
    mut othercaps: gst::Caps,
) -> gst::Caps {
    if direction == gst::PadDirection::Sink && info.is_some() {
        // BaseTransform intersects peer alternatives before fixation. Nested
        // tensor structures can thereby accumulate unrelated decoder groups;
        // retain only candidates matching the groups this transform produces.
        let authoritative = transform_caps(info, direction, caps, None);
        let rejected = othercaps
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                (!authoritative
                    .iter()
                    .any(|expected| same_tensor_groups(candidate, expected)))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        for index in rejected.into_iter().rev() {
            othercaps.make_mut().remove_structure(index);
        }
    }
    othercaps.fixate();
    othercaps
}

fn same_tensor_groups(left: &gst::StructureRef, right: &gst::StructureRef) -> bool {
    match (
        left.get::<gst::Structure>("tensors"),
        right.get::<gst::Structure>("tensors"),
    ) {
        (Ok(left), Ok(right)) => {
            left.n_fields() == right.n_fields() && left.fields().all(|field| right.has_field(field))
        }
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_INFO: &str = include_str!("../tests/fixtures/identity.onnx.modelinfo");

    #[test]
    fn maps_every_supported_output_scalar_to_gstreamer_128() {
        for (scalar, data_type) in [
            (ScalarType::Float16, gst_analytics::TensorDataType::Float16),
            (ScalarType::Float32, gst_analytics::TensorDataType::Float32),
            (ScalarType::Float64, gst_analytics::TensorDataType::Float64),
            (ScalarType::Int8, gst_analytics::TensorDataType::Int8),
            (ScalarType::Int16, gst_analytics::TensorDataType::Int16),
            (ScalarType::Int32, gst_analytics::TensorDataType::Int32),
            (ScalarType::Int64, gst_analytics::TensorDataType::Int64),
            (ScalarType::Uint8, gst_analytics::TensorDataType::Uint8),
            (ScalarType::Uint16, gst_analytics::TensorDataType::Uint16),
            (ScalarType::Uint32, gst_analytics::TensorDataType::Uint32),
            (ScalarType::Uint64, gst_analytics::TensorDataType::Uint64),
        ] {
            assert_eq!(tensor_data_type(scalar), data_type);
        }
    }

    #[test]
    fn transform_caps_strips_reverse_decoder_groups_from_every_structure() {
        gst::init().expect("initializing GStreamer");
        let info = ModelInfo::parse(MODEL_INFO).expect("parsing fixture model-info");

        let mut downstream = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", decoder_groups(&info))
            .build();
        downstream
            .get_mut()
            .expect("mutable caps")
            .append_structure(
                gst::Structure::builder("video/x-raw")
                    .field("format", "BGR")
                    .field("tensors", decoder_groups(&info))
                    .build(),
            );
        let media_filter = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build();
        let reverse = transform_caps(
            Some(&info),
            gst::PadDirection::Src,
            &downstream,
            Some(&media_filter),
        );
        assert_eq!(
            reverse.size(),
            1,
            "the filter must intersect transformed caps"
        );
        for structure in reverse.iter() {
            assert!(!structure.has_field("tensors"));
            assert_eq!(structure.get::<String>("format").unwrap(), "RGB");
            assert_eq!(structure.get::<i32>("width").unwrap(), 2);
            assert_eq!(structure.get::<i32>("height").unwrap(), 1);
        }
    }

    #[test]
    fn fixate_caps_selects_only_the_current_model_group() {
        gst::init().expect("initializing GStreamer");
        let info = ModelInfo::parse(MODEL_INFO).expect("parsing fixture model-info");
        let plain_sink = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build();
        let decoder_filter = decoder_filter(&info);
        let selected = fixate_caps(
            Some(&info),
            gst::PadDirection::Sink,
            &plain_sink,
            decoder_filter,
        );
        let groups = selected
            .structure(0)
            .expect("selected structure")
            .get::<gst::Structure>("tensors")
            .expect("selected tensor groups");
        assert_eq!(groups.n_fields(), 1);
        assert!(!groups.has_field("facedetectortensordecoder"));
        assert_eq!(
            groups
                .get::<gst::UniqueList>(info.group_id())
                .expect("model output group")
                .as_slice()
                .len(),
            2
        );
    }

    #[test]
    fn transform_and_fixate_caps_preserve_genuine_upstream_groups() {
        gst::init().expect("initializing GStreamer");
        let info = ModelInfo::parse(MODEL_INFO).expect("parsing fixture model-info");
        let mut genuine_groups = gst::Structure::new_empty("tensorgroups");
        genuine_groups.set(
            "upstream-model",
            gst::UniqueList::new([synthetic_tensor("upstream-output")]),
        );
        let genuine_sink = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", genuine_groups)
            .build();
        let composed = transform_caps(Some(&info), gst::PadDirection::Sink, &genuine_sink, None);
        let groups = composed
            .structure(0)
            .expect("composed structure")
            .get::<gst::Structure>("tensors")
            .expect("composed tensor groups");
        assert_eq!(groups.n_fields(), 2);
        assert!(groups.has_field("upstream-model"));
        assert!(!groups.has_field("facedetectortensordecoder"));
        assert_eq!(
            groups
                .get::<gst::UniqueList>(info.group_id())
                .expect("composed model output group")
                .as_slice()
                .len(),
            2
        );

        let polluted_sink = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", decoder_groups(&info))
            .build();
        let cleaned = transform_caps(Some(&info), gst::PadDirection::Sink, &polluted_sink, None);
        let cleaned_groups = cleaned
            .structure(0)
            .expect("cleaned structure")
            .get::<gst::Structure>("tensors")
            .expect("cleaned tensor groups");
        assert_eq!(cleaned_groups.n_fields(), 1);
        assert!(cleaned_groups.has_field(info.group_id()));
        assert!(!cleaned_groups.has_field("facedetectortensordecoder"));

        let mut polluted_groups = groups.clone();
        polluted_groups.set(
            "facedetectortensordecoder",
            gst::UniqueList::new([synthetic_tensor("face-output")]),
        );
        let mut candidates = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", polluted_groups)
            .build();
        candidates
            .get_mut()
            .expect("mutable candidates")
            .append_structure(
                gst::Structure::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("tensors", groups)
                    .build(),
            );
        let selected = fixate_caps(
            Some(&info),
            gst::PadDirection::Sink,
            &genuine_sink,
            candidates,
        );
        let groups = selected
            .structure(0)
            .expect("selected composed structure")
            .get::<gst::Structure>("tensors")
            .expect("selected composed tensor groups");
        assert_eq!(groups.n_fields(), 2);
        assert!(groups.has_field("upstream-model"));
        assert!(groups.has_field(info.group_id()));
        assert!(!groups.has_field("facedetectortensordecoder"));
    }

    fn decoder_groups(info: &ModelInfo) -> gst::Structure {
        let mut groups = gst::Structure::new_empty("tensorgroups");
        groups.set(
            "facedetectortensordecoder",
            gst::UniqueList::new([synthetic_tensor("face-output")]),
        );
        groups.set(info.group_id(), model_group(info));
        groups
    }

    fn decoder_filter(info: &ModelInfo) -> gst::Caps {
        let mut filter = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", decoder_groups(info))
            .build();
        let mut model_only = gst::Structure::new_empty("tensorgroups");
        model_only.set(info.group_id(), model_group(info));
        filter.get_mut().expect("mutable caps").append_structure(
            gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("tensors", model_only)
                .build(),
        );
        filter
    }

    fn model_group(info: &ModelInfo) -> gst::UniqueList {
        gst::UniqueList::new(info.outputs().iter().map(tensor_caps).collect::<Vec<_>>())
    }

    fn synthetic_tensor(id: &str) -> gst::Caps {
        gst::Caps::builder("tensor/strided")
            .field("dims", gst::Array::from_values([1_i32.to_send_value()]))
            .field("dims-order", "row-major")
            .field("type", "float32")
            .field("tensor-id", id)
            .build()
    }
}
