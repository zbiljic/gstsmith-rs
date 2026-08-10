use crate::engine::InputTensor;
use crate::model_info::{ScalarType, TensorDescription};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Rgb,
    Bgr,
    Rgba,
    Bgra,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChannelOrder {
    #[default]
    Rgb,
    Bgr,
}

impl ChannelOrder {
    fn destination_channel(self, semantic_channel: usize) -> usize {
        match self {
            Self::Rgb => semantic_channel,
            Self::Bgr => 2 - semantic_channel,
        }
    }
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    fn rgb(self, source: &[u8]) -> Result<[u8; 3], String> {
        let red = source
            .first()
            .copied()
            .ok_or_else(|| "pixel is missing its red/blue component".to_owned())?;
        let green = source
            .get(1)
            .copied()
            .ok_or_else(|| "pixel is missing its green component".to_owned())?;
        let blue = source
            .get(2)
            .copied()
            .ok_or_else(|| "pixel is missing its blue/red component".to_owned())?;
        Ok(match self {
            Self::Rgb | Self::Rgba => [red, green, blue],
            Self::Bgr | Self::Bgra => [blue, green, red],
        })
    }
}

pub fn preprocess(
    source: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    format: PixelFormat,
    channel_order: ChannelOrder,
    input: &TensorDescription,
) -> Result<InputTensor, String> {
    let min_stride = width
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| "input row size overflow".to_owned())?;
    if stride < min_stride {
        return Err("video stride is shorter than a pixel row".to_owned());
    }
    let source_len = stride
        .checked_mul(height)
        .ok_or_else(|| "video size overflow".to_owned())?;
    if source.len() < source_len {
        return Err("mapped video frame is shorter than its declared stride".to_owned());
    }
    let channels_first = input.dims.get(1) == Some(&3);
    let expected = if channels_first {
        [1, 3, height, width]
    } else {
        [1, height, width, 3]
    };
    if input.dims != expected {
        return Err(format!(
            "video dimensions {width}x{height} do not match model input {:?}",
            input.dims
        ));
    }
    let values = width
        .checked_mul(height)
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(|| "input tensor size overflow".to_owned())?;
    match input.data_type {
        ScalarType::Uint8 => {
            let mut packed = vec![0; values];
            copy_pixels(
                source,
                stride,
                width,
                height,
                format,
                |index, channel, value| {
                    let destination_channel = channel_order.destination_channel(channel);
                    let destination = if channels_first {
                        destination_channel * width * height + index
                    } else {
                        index * 3 + destination_channel
                    };
                    let target = packed
                        .get_mut(destination)
                        .ok_or_else(|| "preprocessor output index overflow".to_owned())?;
                    *target = value;
                    Ok(())
                },
            )?;
            Ok(InputTensor::Uint8(packed))
        }
        ScalarType::Float32 => {
            let mut packed = vec![0.0; values];
            copy_pixels(
                source,
                stride,
                width,
                height,
                format,
                |index, channel, value| {
                    let range_index = if input.ranges.len() == 1 { 0 } else { channel };
                    let range = input.ranges.get(range_index).ok_or_else(|| {
                        "model-info ranges do not describe every channel".to_owned()
                    })?;
                    let destination_channel = channel_order.destination_channel(channel);
                    let destination = if channels_first {
                        destination_channel * width * height + index
                    } else {
                        index * 3 + destination_channel
                    };
                    let target = packed
                        .get_mut(destination)
                        .ok_or_else(|| "preprocessor output index overflow".to_owned())?;
                    *target = range.0 + f32::from(value) * (range.1 - range.0) / 255.0;
                    Ok(())
                },
            )?;
            Ok(InputTensor::Float32(packed))
        }
        _ => Err("model-info permits only float32 or uint8 image inputs".to_owned()),
    }
}

fn copy_pixels(
    source: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    format: PixelFormat,
    mut write: impl FnMut(usize, usize, u8) -> Result<(), String>,
) -> Result<(), String> {
    let row_bytes = width
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| "input row size overflow".to_owned())?;
    for (row_index, row) in source.chunks_exact(stride).take(height).enumerate() {
        let pixels = row
            .get(..row_bytes)
            .ok_or_else(|| "video row is shorter than declared stride".to_owned())?;
        for (column, pixel) in pixels.chunks_exact(format.bytes_per_pixel()).enumerate() {
            let rgb = format.rgb(pixel)?;
            let index = row_index
                .checked_mul(width)
                .and_then(|value| value.checked_add(column))
                .ok_or_else(|| "pixel index overflow".to_owned())?;
            for (channel, value) in rgb.into_iter().enumerate() {
                write(index, channel, value)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model_info::{DimOrder, ScalarType, TensorDescription};

    use super::{ChannelOrder, InputTensor, PixelFormat, preprocess};

    fn input(dims: Vec<usize>) -> TensorDescription {
        TensorDescription {
            name: "input".to_owned(),
            id: "input".to_owned(),
            data_type: ScalarType::Float32,
            dims,
            dim_order: DimOrder::RowMajor,
            ranges: vec![(0.0, 1.0)],
        }
    }

    fn float_values(result: Result<InputTensor, String>) -> Vec<f32> {
        match result {
            Ok(InputTensor::Float32(values)) => values,
            Ok(InputTensor::Uint8(_)) => panic!("expected float32 preprocessing output"),
            Err(error) => panic!("preprocessing failed: {error}"),
        }
    }

    fn uint8_values(result: Result<InputTensor, String>) -> Vec<u8> {
        match result {
            Ok(InputTensor::Uint8(values)) => values,
            Ok(InputTensor::Float32(_)) => panic!("expected uint8 preprocessing output"),
            Err(error) => panic!("preprocessing failed: {error}"),
        }
    }

    #[test]
    fn packs_stride_aware_hwc_rgb() {
        let pixels = [10, 20, 30, 40, 50, 60, 99, 99];
        let processed = preprocess(
            &pixels,
            8,
            2,
            1,
            PixelFormat::Rgb,
            ChannelOrder::Rgb,
            &input(vec![1, 1, 2, 3]),
        );
        assert!(processed.is_ok());
        let Some(result) = processed.ok() else {
            return;
        };
        let values = match result {
            InputTensor::Float32(values) => values,
            InputTensor::Uint8(_) => Vec::new(),
        };
        assert_eq!(
            values,
            [
                10.0 / 255.0,
                20.0 / 255.0,
                30.0 / 255.0,
                40.0 / 255.0,
                50.0 / 255.0,
                60.0 / 255.0
            ]
        );
    }

    #[test]
    fn packs_bgr_to_chw() {
        let pixels = [30, 20, 10, 60, 50, 40];
        let processed = preprocess(
            &pixels,
            6,
            2,
            1,
            PixelFormat::Bgr,
            ChannelOrder::Rgb,
            &input(vec![1, 3, 1, 2]),
        );
        assert!(processed.is_ok());
        let Some(result) = processed.ok() else {
            return;
        };
        let values = match result {
            InputTensor::Float32(values) => values,
            InputTensor::Uint8(_) => Vec::new(),
        };
        assert_eq!(
            values,
            [
                10.0 / 255.0,
                40.0 / 255.0,
                20.0 / 255.0,
                50.0 / 255.0,
                30.0 / 255.0,
                60.0 / 255.0
            ]
        );
    }

    #[test]
    fn packs_alpha_formats_without_reading_alpha() {
        let rgba = preprocess(
            &[10, 20, 30, 99],
            4,
            1,
            1,
            PixelFormat::Rgba,
            ChannelOrder::Rgb,
            &input(vec![1, 1, 1, 3]),
        );
        let bgra = preprocess(
            &[30, 20, 10, 99],
            4,
            1,
            1,
            PixelFormat::Bgra,
            ChannelOrder::Rgb,
            &input(vec![1, 1, 1, 3]),
        );
        let rgba = rgba.ok();
        let bgra = bgra.ok();
        match (rgba, bgra) {
            (Some(InputTensor::Float32(rgba)), Some(InputTensor::Float32(bgra))) => {
                assert_eq!(rgba, bgra);
            }
            _ => panic!("alpha formats did not produce float inputs"),
        }
    }

    #[test]
    fn packs_source_formats_for_explicit_model_channel_orders() {
        struct Case {
            source: &'static [u8],
            stride: usize,
            format: PixelFormat,
            order: ChannelOrder,
            dims: Vec<usize>,
            expected: &'static [f32],
        }

        let cases = [
            Case {
                source: &[10, 20, 30],
                stride: 3,
                format: PixelFormat::Rgb,
                order: ChannelOrder::Rgb,
                dims: vec![1, 1, 1, 3],
                expected: &[10.0, 20.0, 30.0],
            },
            Case {
                source: &[30, 20, 10],
                stride: 3,
                format: PixelFormat::Bgr,
                order: ChannelOrder::Rgb,
                dims: vec![1, 1, 1, 3],
                expected: &[10.0, 20.0, 30.0],
            },
            Case {
                source: &[10, 20, 30],
                stride: 3,
                format: PixelFormat::Rgb,
                order: ChannelOrder::Bgr,
                dims: vec![1, 1, 1, 3],
                expected: &[30.0, 20.0, 10.0],
            },
            Case {
                source: &[30, 20, 10],
                stride: 3,
                format: PixelFormat::Bgr,
                order: ChannelOrder::Bgr,
                dims: vec![1, 3, 1, 1],
                expected: &[30.0, 20.0, 10.0],
            },
            Case {
                source: &[10, 20, 30, 77, 40, 50, 60, 88],
                stride: 8,
                format: PixelFormat::Rgba,
                order: ChannelOrder::Bgr,
                dims: vec![1, 1, 2, 3],
                expected: &[30.0, 20.0, 10.0, 60.0, 50.0, 40.0],
            },
            Case {
                source: &[30, 20, 10, 77, 99, 99],
                stride: 6,
                format: PixelFormat::Bgra,
                order: ChannelOrder::Bgr,
                dims: vec![1, 3, 1, 1],
                expected: &[30.0, 20.0, 10.0],
            },
        ];

        for case in cases {
            let mut description = input(case.dims);
            description.ranges = vec![(0.0, 255.0)];
            let processed = preprocess(
                case.source,
                case.stride,
                if description.dims.get(1) == Some(&3) {
                    description.dims[3]
                } else {
                    description.dims[2]
                },
                1,
                case.format,
                case.order,
                &description,
            );
            assert_eq!(float_values(processed), case.expected);
        }
    }

    #[test]
    fn applies_semantic_ranges_before_bgr_placement() {
        let mut description = input(vec![1, 1, 1, 3]);
        description.ranges = vec![(0.0, 255.0), (100.0, 355.0), (200.0, 455.0)];
        let processed = preprocess(
            &[10, 20, 30],
            3,
            1,
            1,
            PixelFormat::Rgb,
            ChannelOrder::Bgr,
            &description,
        );
        assert_eq!(float_values(processed), [230.0, 120.0, 10.0]);
    }

    #[test]
    fn reorders_uint8_for_bgr_models() {
        let mut description = input(vec![1, 3, 1, 2]);
        description.data_type = ScalarType::Uint8;
        let processed = preprocess(
            &[10, 20, 30, 40, 50, 60],
            6,
            2,
            1,
            PixelFormat::Rgb,
            ChannelOrder::Bgr,
            &description,
        );
        assert_eq!(uint8_values(processed), [30, 60, 20, 50, 10, 40]);
    }
}
