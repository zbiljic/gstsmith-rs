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
            if let Some(info) = info
                && let Ok(mut groups) = structure.get::<gst::Structure>("tensors")
            {
                groups.remove_field(info.group_id());
                structure.set("tensors", groups);
            }
        } else if let Some(info) = info {
            let mut groups = structure
                .get::<gst::Structure>("tensors")
                .unwrap_or_else(|_| gst::Structure::new_empty("tensorgroups"));
            let descriptors = info.outputs().iter().map(tensor_caps).collect::<Vec<_>>();
            groups.set(info.group_id(), gst::UniqueList::new(descriptors));
            structure.set("tensors", groups);
        }
    }
    filter.map_or(result.clone(), |filter| {
        filter.intersect_with_mode(&result, gst::CapsIntersectMode::First)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
