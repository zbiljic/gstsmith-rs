use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarType {
    Float16,
    Float64,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

impl ScalarType {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "float16" => Ok(Self::Float16),
            "float64" => Ok(Self::Float64),
            "float32" => Ok(Self::Float32),
            "int8" => Ok(Self::Int8),
            "int16" => Ok(Self::Int16),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "uint8" => Ok(Self::Uint8),
            "uint16" => Ok(Self::Uint16),
            "uint32" => Ok(Self::Uint32),
            "uint64" => Ok(Self::Uint64),
            _ => Err(format!(
                "unsupported scalar type {value:?}; supported types are float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, and uint64"
            )),
        }
    }

    #[must_use]
    pub fn is_supported_input(self) -> bool {
        matches!(self, Self::Float32 | Self::Uint8)
    }

    /// Bytes per element.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 => 1,
            Self::Float16 | Self::Int16 | Self::Uint16 => 2,
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 => 8,
        }
    }

    #[must_use]
    pub fn as_caps_name(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float64 => "float64",
            Self::Float32 => "float32",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DimOrder {
    RowMajor,
    ColMajor,
}

impl DimOrder {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("row-major") {
            "row-major" => Ok(Self::RowMajor),
            "col-major" => Ok(Self::ColMajor),
            value => Err(format!("unsupported dims-order {value:?}")),
        }
    }

    #[must_use]
    pub fn as_caps_name(self) -> &'static str {
        match self {
            Self::RowMajor => "row-major",
            Self::ColMajor => "col-major",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorDescription {
    pub name: String,
    pub id: String,
    pub data_type: ScalarType,
    pub dims: Vec<usize>,
    pub dim_order: DimOrder,
    pub ranges: Vec<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    group_id: String,
    input: TensorDescription,
    outputs: Vec<TensorDescription>,
}

/// Which kind of model a model-info describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelKind {
    /// One image input packed from video frames (`ranges` required).
    Image,
    /// Inputs taken from upstream tensors, already preprocessed.
    Tensor,
}

/// The sections of a model-info file, split by role.
struct ParsedModel {
    group_id: String,
    inputs: Vec<TensorDescription>,
    outputs: Vec<TensorDescription>,
}

fn parse_model(contents: &str, kind: ModelKind) -> Result<ParsedModel, String> {
    let sections = parse_sections(contents)?;
    let header = sections
        .iter()
        .find(|section| section.name == "modelinfo")
        .ok_or_else(|| "model-info requires a [modelinfo] section".to_owned())?;
    let version = require(header, "version", "modelinfo")?;
    if version != "1.0" {
        return Err(format!(
            "unsupported model-info version {version:?}; only 1.0 is supported"
        ));
    }
    let group_id = non_empty(require(header, "group-id", "modelinfo")?, "group-id")?.to_owned();
    let mut ids = BTreeSet::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for section in sections
        .iter()
        .filter(|section| section.name != "modelinfo")
    {
        let tensor = parse_tensor(section, kind)?;
        if !ids.insert(tensor.description.id.clone()) {
            return Err(format!("duplicate tensor id {:?}", tensor.description.id));
        }
        match tensor.direction {
            Direction::Output => outputs.push(tensor.description),
            Direction::Input => inputs.push(tensor.description),
        }
    }
    if outputs.is_empty() {
        return Err("at least one output is required".to_owned());
    }
    Ok(ParsedModel {
        group_id,
        inputs,
        outputs,
    })
}

impl ModelInfo {
    pub fn parse(contents: &str) -> Result<Self, String> {
        let parsed = parse_model(contents, ModelKind::Image)?;
        if parsed.inputs.len() != 1 {
            return Err(format!(
                "exactly one image input is required, found {}",
                parsed.inputs.len()
            ));
        }
        let input = parsed
            .inputs
            .into_iter()
            .next()
            .ok_or_else(|| "missing input".to_owned())?;
        validate_image_input(&input)?;
        Ok(Self {
            group_id: parsed.group_id,
            input,
            outputs: parsed.outputs,
        })
    }

    /// The tensor group this model produces and the image size it requires.
    #[must_use]
    pub fn caps_contract(&self) -> CapsContract<'_> {
        CapsContract {
            group_id: &self.group_id,
            outputs: &self.outputs,
            image: self.image_dimensions().ok(),
        }
    }

    #[must_use]
    pub fn group_id(&self) -> &str {
        &self.group_id
    }
    #[must_use]
    pub fn input(&self) -> &TensorDescription {
        &self.input
    }

    /// Outputs to request and attach, in model-info order. A model may have
    /// more outputs; undeclared ones are not published.
    #[must_use]
    pub fn outputs(&self) -> &[TensorDescription] {
        &self.outputs
    }

    pub fn image_dimensions(&self) -> Result<(usize, usize), String> {
        image_layout(&self.input.dims)
            .map(|layout| (layout.width, layout.height))
            .ok_or_else(|| "input dimensions do not describe a static RGB image".to_owned())
    }
}

/// The tensor group an element produces, and the image size (width, height)
/// its input video must have, for image models.
#[derive(Clone, Copy, Debug)]
pub struct CapsContract<'a> {
    pub group_id: &'a str,
    pub outputs: &'a [TensorDescription],
    pub image: Option<(usize, usize)>,
}

/// Model-info of a tensor-input model: every input is taken,
/// by tensor id, from upstream `GstTensorMeta`.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorModelInfo {
    group_id: String,
    inputs: Vec<TensorDescription>,
    outputs: Vec<TensorDescription>,
}

impl TensorModelInfo {
    pub fn parse(contents: &str) -> Result<Self, String> {
        let parsed = parse_model(contents, ModelKind::Tensor)?;
        if parsed.inputs.is_empty() {
            return Err("at least one input is required".to_owned());
        }
        Ok(Self {
            group_id: parsed.group_id,
            inputs: parsed.inputs,
            outputs: parsed.outputs,
        })
    }

    #[must_use]
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Inputs read from upstream tensors, in model-info order.
    #[must_use]
    pub fn inputs(&self) -> &[TensorDescription] {
        &self.inputs
    }

    /// Outputs to request and attach, in model-info order.
    #[must_use]
    pub fn outputs(&self) -> &[TensorDescription] {
        &self.outputs
    }

    /// The tensor group this model produces; there is no image constraint.
    #[must_use]
    pub fn caps_contract(&self) -> CapsContract<'_> {
        CapsContract {
            group_id: &self.group_id,
            outputs: &self.outputs,
            image: None,
        }
    }
}

struct ParsedTensor {
    direction: Direction,
    description: TensorDescription,
}

fn parse_tensor(section: &Section, kind: ModelKind) -> Result<ParsedTensor, String> {
    let name = section.name.as_str();
    let direction = match require(section, "dir", name)? {
        "input" => Direction::Input,
        "output" => Direction::Output,
        value => return Err(format!("tensor {name:?} has invalid dir {value:?}")),
    };
    let id = non_empty(require(section, "id", name)?, "id")?.to_owned();
    let dims = parse_dims(require(section, "dims", name)?)?;
    let data_type = ScalarType::parse(require(section, "type", name)?)?;
    dims.iter()
        .try_fold(data_type.size(), |size, dimension| {
            size.checked_mul(*dimension)
        })
        .and_then(|size| isize::try_from(size).ok())
        .ok_or_else(|| {
            format!("tensor {name:?} byte size exceeds the platform allocation limit")
        })?;
    let dim_order = DimOrder::parse(section.values.get("dims-order").map(String::as_str))?;
    if kind == ModelKind::Tensor && dim_order != DimOrder::RowMajor {
        return Err(format!(
            "tensor {name:?} must use row-major dimension order"
        ));
    }
    let ranges = if direction == Direction::Input && kind == ModelKind::Image {
        if !data_type.is_supported_input() {
            return Err(format!(
                "input tensor {name:?} type {} is unsupported; image inputs must be float32 or uint8",
                data_type.as_caps_name()
            ));
        }
        parse_ranges(require(section, "ranges", name)?)?
    } else {
        Vec::new()
    };
    Ok(ParsedTensor {
        direction,
        description: TensorDescription {
            name: section.name.clone(),
            id,
            data_type,
            dims,
            dim_order,
            ranges,
        },
    })
}

#[derive(Debug)]
struct Section {
    name: String,
    values: BTreeMap<String, String>,
}

fn parse_sections(contents: &str) -> Result<Vec<Section>, String> {
    // GstAnalyticsModelInfo uses GKeyFile too. Its API does not enumerate
    // tensor sections or expose the declared dimensions needed by our backends.
    let key_file = gst::glib::KeyFile::new();
    key_file
        .load_from_data(contents, gst::glib::KeyFileFlags::NONE)
        .map_err(|error| format!("invalid model-info: {error}"))?;
    key_file
        .groups()
        .iter()
        .map(|name| {
            let keys = key_file.keys(name).map_err(|error| error.to_string())?;
            let values = keys
                .iter()
                .filter(|key| {
                    matches!(
                        key.as_str(),
                        "version"
                            | "group-id"
                            | "dir"
                            | "id"
                            | "type"
                            | "dims"
                            | "dims-order"
                            | "ranges"
                    )
                })
                .map(|key| {
                    key_file
                        .string(name, key)
                        .map(|value| (key.to_string(), value.to_string()))
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<_, _>>()?;
            Ok(Section {
                name: name.to_string(),
                values,
            })
        })
        .collect()
}

fn require<'a>(section: &'a Section, key: &str, section_name: &str) -> Result<&'a str, String> {
    section
        .values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("section {section_name:?} is missing {key:?}"))
}

fn non_empty<'a>(value: &'a str, name: &str) -> Result<&'a str, String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(value.trim())
    }
}

fn parse_dims(value: &str) -> Result<Vec<usize>, String> {
    let dims: Result<Vec<_>, _> = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_error| format!("invalid static dimension {part:?}"))
        })
        .collect();
    let dims = dims?;
    if dims.is_empty() {
        return Err("dimensions must not be empty".to_owned());
    }
    if dims.contains(&0) || dims.iter().any(|dimension| *dimension > i32::MAX as usize) {
        return Err(
            "dimensions must be positive, static, and representable by tensor caps".to_owned(),
        );
    }
    Ok(dims)
}

fn parse_ranges(value: &str) -> Result<Vec<(f32, f32)>, String> {
    let ranges: Result<Vec<_>, _> = value
        .split(';')
        .map(|range| {
            let (low, high) = range
                .split_once(',')
                .ok_or_else(|| format!("invalid range {range:?}"))?;
            let low = low
                .trim()
                .parse::<f32>()
                .map_err(|_error| format!("invalid range lower bound {low:?}"))?;
            let high = high
                .trim()
                .parse::<f32>()
                .map_err(|_error| format!("invalid range upper bound {high:?}"))?;
            if !low.is_finite() || !high.is_finite() {
                return Err("ranges must be finite".to_owned());
            }
            Ok((low, high))
        })
        .collect();
    let ranges = ranges?;
    if ranges.is_empty() {
        return Err("input ranges must not be empty".to_owned());
    }
    Ok(ranges)
}

/// Whether a model's tensor shape accepts the model-info dimensions: ranks
/// agree, dynamic model dimensions (`None`) accept any declared size, and
/// fixed ones must be equal. Model-info binds the dynamic dimensions.
#[must_use]
pub fn dims_match(model: &[Option<usize>], declared: &[usize]) -> bool {
    model.len() == declared.len()
        && model
            .iter()
            .zip(declared)
            .all(|(model, declared)| model.is_none_or(|model| model == *declared))
}

/// How an image input packs one RGB frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageLayout {
    pub channels_first: bool,
    pub height: usize,
    pub width: usize,
}

/// The layout of image input `dims`: unit leading dimensions (batch first,
/// then any other unit dimensions such as a frame count) followed by
/// `3,H,W` or `H,W,3`.
#[must_use]
pub fn image_layout(dims: &[usize]) -> Option<ImageLayout> {
    let split = dims.len().checked_sub(3)?;
    let (leading, image) = dims.split_at(split);
    if leading.is_empty() || leading.iter().any(|dimension| *dimension != 1) {
        return None;
    }
    match image {
        [3, height, width] if *width != 3 => Some(ImageLayout {
            channels_first: true,
            height: *height,
            width: *width,
        }),
        [height, width, 3] if *height != 3 => Some(ImageLayout {
            channels_first: false,
            height: *height,
            width: *width,
        }),
        _ => None,
    }
}

fn validate_image_input(input: &TensorDescription) -> Result<(), String> {
    if input.dims.first() != Some(&1) {
        return Err(format!(
            "image input {:?} must have a static batch dimension of one",
            input.name
        ));
    }
    if image_layout(&input.dims).is_none() {
        return Err(
            "input dimensions must be unit leading dimensions followed by an unambiguous 3,H,W or H,W,3 image"
                .to_owned(),
        );
    }
    if input.ranges.len() != 1 && input.ranges.len() != 3 {
        return Err("input ranges must contain one or three channel ranges".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "[modelinfo]\nversion=1.0\ngroup-id=test-group\n\n[input]\nid=image\ntype=float32\ndims=1,2,3,3\ndir=input\nranges=0,255;0,255;0,255\n\n[first]\nid=first-output\ntype=float32\ndims=1,18\ndir=output\n\n[second]\nid=second-output\ntype=uint8\ndims=1,2,3,3\ndir=output\ndims-order=col-major\n";

    #[test]
    fn preserves_order_and_defaults() {
        let parsed = ModelInfo::parse(VALID);
        assert!(parsed.is_ok());
        let Some(info) = parsed.ok() else {
            return;
        };
        assert_eq!(info.group_id(), "test-group");
        assert_eq!(info.input().dim_order, DimOrder::RowMajor);
        assert_eq!(info.outputs()[0].id, "first-output");
        assert_eq!(info.outputs()[1].dim_order, DimOrder::ColMajor);
    }

    #[test]
    fn accepts_chw_image_input() {
        let parsed = ModelInfo::parse(&VALID.replacen("1,2,3,3", "1,3,2,4", 1));
        assert!(parsed.is_ok());
        let Some(info) = parsed.ok() else {
            return;
        };
        assert_eq!(info.input().dims, [1, 3, 2, 4]);
    }

    #[test]
    fn accepts_float16_and_int32_outputs_but_not_int32_inputs()
    -> Result<(), Box<dyn std::error::Error>> {
        for scalar_type in ["float16", "int32"] {
            let replacement = format!("type={scalar_type}");
            if let Err(error) = ModelInfo::parse(&VALID.replace("type=uint8", &replacement)) {
                return Err(std::io::Error::other(format!(
                    "{scalar_type} output was rejected: {error}"
                ))
                .into());
            }
        }
        let input = ModelInfo::parse(&VALID.replace("type=float32", "type=int32"));
        if input.is_ok() {
            return Err(std::io::Error::other("int32 input was accepted").into());
        }
        Ok(())
    }

    #[test]
    fn accepts_leading_unit_dimensions_before_the_image() {
        let info = ModelInfo::parse(&VALID.replacen("1,2,3,3", "1,1,3,2,4", 1))
            .expect("five-dimensional image input parses");
        assert_eq!(info.image_dimensions(), Ok((4, 2)));
        assert_eq!(
            image_layout(&[1, 1, 2, 4, 3]),
            Some(ImageLayout {
                channels_first: false,
                height: 2,
                width: 4
            })
        );
        for invalid in [&[1, 2, 3, 4, 4][..], &[3, 4, 4], &[1, 3, 4, 3]] {
            assert_eq!(image_layout(invalid), None, "{invalid:?}");
        }
    }

    #[test]
    fn declared_dimensions_bind_dynamic_model_dimensions() {
        assert!(dims_match(&[None, None, Some(3)], &[1, 400, 3]));
        assert!(!dims_match(&[None, Some(768)], &[1, 400]));
        assert!(!dims_match(&[None, None], &[1, 2, 3]));
    }

    const TENSOR_MODEL: &str = "[modelinfo]\nversion=1.0\ngroup-id=tensor-group\n\n[input_embs]\nid=embeddings\ntype=float32\ndims=1,400,768\ndir=input\n\n[attention_mask]\nid=mask\ntype=uint8\ndims=1,400\ndir=input\n\n[scale]\nid=scale\ntype=float32\ndims=1\ndir=input\n\n[scores]\nid=scores\ntype=float32\ndims=1,400,2\ndir=output\n";

    #[test]
    fn parses_tensor_models_with_several_inputs() {
        let info = TensorModelInfo::parse(TENSOR_MODEL).expect("tensor model parses");
        assert_eq!(info.group_id(), "tensor-group");
        let inputs = info
            .inputs()
            .iter()
            .map(|i| (i.id.as_str(), i.data_type))
            .collect::<Vec<_>>();
        assert_eq!(
            inputs,
            [
                ("embeddings", ScalarType::Float32),
                ("mask", ScalarType::Uint8),
                ("scale", ScalarType::Float32)
            ]
        );
        assert_eq!(info.outputs()[0].dims, [1, 400, 2]);
        assert_eq!(info.caps_contract().image, None);
    }

    #[test]
    fn ignores_unknown_fields() {
        let expected = TensorModelInfo::parse(TENSOR_MODEL).expect("tensor model parses");
        let contents =
            TENSOR_MODEL.replace("dir=input", "dir=input\nunknown=ignored\\q\nranges=0,1");
        let actual = TensorModelInfo::parse(&contents).expect("extra metadata parses");
        assert_eq!(actual, expected);
    }

    #[test]
    fn uses_glib_key_file_escaping_and_header_placement() {
        let contents = TENSOR_MODEL
            .replace("[modelinfo]\nversion=1.0\ngroup-id=tensor-group\n\n", "")
            .replace("id=embeddings", "id=embedding\\sspace")
            + "\n[modelinfo]\nversion=1.0\ngroup-id=tensor-group\n";
        let info = TensorModelInfo::parse(&contents).expect("GLib key file parses");
        assert_eq!(info.inputs()[0].id, "embedding space");
        assert_eq!(info.outputs()[0].id, "scores");
    }

    #[test]
    fn rejects_unsupported_tensor_types_without_conversion() {
        for data_type in ["bool", "int4", "uint4", "bfloat16"] {
            let contents = TENSOR_MODEL.replace("type=uint8", &format!("type={data_type}"));
            let error = TensorModelInfo::parse(&contents).expect_err("unsupported type");
            assert!(error.contains("unsupported scalar type"), "{error}");
        }
    }

    #[test]
    fn rejects_invalid_tensor_models() {
        for invalid in [
            TENSOR_MODEL.replace("dims=1,400,768", "dims="),
            TENSOR_MODEL.replace("dims=1,400,768", "dims=0,400,768"),
            TENSOR_MODEL.replace("dims=1,400,768", "dims=-1,400,768"),
            TENSOR_MODEL.replace("dims=1,400,768", "dims=2147483648"),
            TENSOR_MODEL.replace("dims=1,400,768", "dims=1,2147483647,2147483647,2147483647"),
            TENSOR_MODEL.replace("dims=1,400,768", "dims=1,2147483647,2147483647"),
            TENSOR_MODEL.replacen("dir=input", "dir=input\ndims-order=col-major", 1),
            TENSOR_MODEL.replace("dir=output", "dir=output\ndims-order=col-major"),
            "[modelinfo]\nversion=1.0\ngroup-id=g\n\n[scores]\nid=scores\ntype=float32\ndims=1,2\ndir=output\n".to_owned(),
        ] {
            assert!(
                TensorModelInfo::parse(&invalid).is_err(),
                "accepted invalid tensor model-info:\n{invalid}"
            );
        }
        let image = ModelInfo::parse(VALID).expect("image model-info parses");
        assert_eq!(image.caps_contract().image, Some((3, 2)));
    }

    #[test]
    fn requires_exactly_one_image_input() {
        let two_images = VALID.to_owned()
            + "\n[other]\nid=other\ntype=float32\ndims=1,3,2,2\ndir=input\nranges=0,1\n";
        let error = ModelInfo::parse(&two_images).expect_err("two image inputs are rejected");
        assert!(error.contains("exactly one image"), "{error}");
    }

    #[test]
    fn rejects_malformed_essential_fields() {
        for invalid in [
            VALID.replace("version=1.0", "version=1.1"),
            VALID.replace("version=1.0", "version=2.0"),
            VALID.replace("group-id=test-group", "group-id="),
            VALID.replace("dir=input", "dir=sideways"),
            VALID.replace("dims=1,18", "dims=1,-1"),
            VALID.replacen("dims=1,2,3,3", "dims=2,2,3,3", 1),
            VALID.replacen("dims=1,2,3,3", "dims=1,3,2,3", 1),
            VALID.replacen("dims=1,2,3,3", "dims=2,3,3", 1),
            VALID.replace("dims-order=col-major", "dims-order=unknown"),
            VALID.replace("id=second-output", "id=first-output"),
            VALID.replace("ranges=0,255;0,255;0,255", "ranges=0,1;0,1"),
        ] {
            if ModelInfo::parse(&invalid).is_ok() {
                assert!(invalid.is_empty(), "parser accepted malformed model-info");
            }
        }
    }
}
