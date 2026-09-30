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
    Bool,
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
            "bool" => Ok(Self::Bool),
            _ => Err(format!(
                "unsupported scalar type {value:?}; supported types are float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, and bool"
            )),
        }
    }

    #[must_use]
    pub fn is_supported_input(self) -> bool {
        matches!(self, Self::Float32 | Self::Uint8)
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
            Self::Bool => "bool",
        }
    }
}

/// The value an element supplies for a constant model input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstantValue {
    Bool(bool),
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    Uint8(u8),
}

impl ConstantValue {
    fn parse(value: &str, data_type: ScalarType) -> Result<Self, String> {
        let invalid = || format!("invalid {} constant {value:?}", data_type.as_caps_name());
        let finite = |parsed: f64| parsed.is_finite().then_some(parsed).ok_or_else(invalid);
        match data_type {
            ScalarType::Bool => match value {
                "true" | "1" => Ok(Self::Bool(true)),
                "false" | "0" => Ok(Self::Bool(false)),
                _ => Err(invalid()),
            },
            ScalarType::Float32 => {
                let parsed = value.parse::<f32>().map_err(|_error| invalid())?;
                finite(f64::from(parsed)).map(|_| Self::Float32(parsed))
            }
            ScalarType::Float64 => value
                .parse::<f64>()
                .map_err(|_error| invalid())
                .and_then(finite)
                .map(Self::Float64),
            ScalarType::Int32 => value.parse().map(Self::Int32).map_err(|_error| invalid()),
            ScalarType::Int64 => value.parse().map(Self::Int64).map_err(|_error| invalid()),
            ScalarType::Uint8 => value.parse().map(Self::Uint8).map_err(|_error| invalid()),
            other => Err(format!(
                "constant inputs of type {} are unsupported; use bool, float32, float64, int32, int64, or uint8",
                other.as_caps_name()
            )),
        }
    }
}

/// A model input the element fills with one value on every run.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantInput {
    pub description: TensorDescription,
    pub value: ConstantValue,
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
    constants: Vec<ConstantInput>,
    outputs: Vec<TensorDescription>,
}

impl ModelInfo {
    pub fn parse(contents: &str) -> Result<Self, String> {
        let sections = parse_sections(contents)?;
        let header = sections
            .first()
            .filter(|section| section.name == "modelinfo")
            .ok_or_else(|| "model-info must start with a [modelinfo] section".to_owned())?;
        require(header, "version", "modelinfo").and_then(|version| {
            if version == "1.0" {
                Ok(())
            } else {
                Err(format!("unsupported model-info version {version:?}"))
            }
        })?;
        let group_id = non_empty(require(header, "group-id", "modelinfo")?, "group-id")?.to_owned();
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut images = Vec::new();
        let mut constants = Vec::new();
        let mut outputs = Vec::new();
        for section in sections.iter().skip(1) {
            if !names.insert(section.name.clone()) {
                return Err(format!("duplicate tensor {:?}", section.name));
            }
            let tensor = parse_tensor(section)?;
            if !ids.insert(tensor.description.id.clone()) {
                return Err(format!("duplicate tensor id {:?}", tensor.description.id));
            }
            match (tensor.direction, tensor.constant) {
                (Direction::Output, _) => outputs.push(tensor.description),
                (Direction::Input, Some(value)) => constants.push(ConstantInput {
                    description: tensor.description,
                    value,
                }),
                (Direction::Input, None) => images.push(tensor.description),
            }
        }
        if images.len() != 1 {
            return Err(format!(
                "exactly one non-constant (image) input is required, found {}",
                images.len()
            ));
        }
        let input = images
            .into_iter()
            .next()
            .ok_or_else(|| "missing input".to_owned())?;
        validate_image_input(&input)?;
        if outputs.is_empty() {
            return Err("at least one output is required".to_owned());
        }
        Ok(Self {
            group_id,
            input,
            constants,
            outputs,
        })
    }

    #[must_use]
    pub fn group_id(&self) -> &str {
        &self.group_id
    }
    #[must_use]
    pub fn input(&self) -> &TensorDescription {
        &self.input
    }
    /// Inputs the element supplies itself, in model-info order.
    #[must_use]
    pub fn constants(&self) -> &[ConstantInput] {
        &self.constants
    }

    /// Outputs to compute and attach, in model-info order. A model may have
    /// more outputs; undeclared ones are not computed.
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

struct ParsedTensor {
    direction: Direction,
    description: TensorDescription,
    constant: Option<ConstantValue>,
}

fn parse_tensor(section: &Section) -> Result<ParsedTensor, String> {
    let name = section.name.as_str();
    let direction = match require(section, "dir", name)? {
        "input" => Direction::Input,
        "output" => Direction::Output,
        value => return Err(format!("tensor {name:?} has invalid dir {value:?}")),
    };
    let id = non_empty(require(section, "id", name)?, "id")?.to_owned();
    let dims = parse_dims(require(section, "dims", name)?)?;
    if dims.first() != Some(&1) {
        return Err(format!(
            "tensor {name:?} must have a static batch dimension of one"
        ));
    }
    let data_type = ScalarType::parse(require(section, "type", name)?)?;
    let constant = section.values.get("constant");
    let ranges = match (direction, constant) {
        (Direction::Input, None) => {
            if !data_type.is_supported_input() {
                return Err(format!(
                    "input tensor {name:?} type {} is unsupported; image inputs must be float32 or uint8",
                    data_type.as_caps_name()
                ));
            }
            parse_ranges(require(section, "ranges", name)?)?
        }
        (Direction::Input, Some(_)) if section.values.contains_key("ranges") => {
            return Err(format!("constant input {name:?} must not declare ranges"));
        }
        (Direction::Output, Some(_)) => {
            return Err(format!("output {name:?} must not declare a constant"));
        }
        _ => Vec::new(),
    };
    Ok(ParsedTensor {
        direction,
        constant: constant
            .map(|value| ConstantValue::parse(value, data_type))
            .transpose()?,
        description: TensorDescription {
            name: section.name.clone(),
            id,
            data_type,
            dims,
            dim_order: DimOrder::parse(section.values.get("dims-order").map(String::as_str))?,
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
    let mut sections = Vec::new();
    let mut current: Option<Section> = None;
    for (line_no, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|name| name.strip_suffix(']'))
        {
            if let Some(section) = current.take() {
                sections.push(section);
            }
            current = Some(Section {
                name: non_empty(name, "section name")?.to_owned(),
                values: BTreeMap::new(),
            });
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {} is not a key=value entry", line_no + 1))?;
        let section = current
            .as_mut()
            .ok_or_else(|| format!("line {} appears before any section", line_no + 1))?;
        let key = non_empty(key.trim(), "key")?.to_owned();
        if section
            .values
            .insert(key.clone(), value.trim().to_owned())
            .is_some()
        {
            return Err(format!(
                "duplicate key {key:?} in section {:?}",
                section.name
            ));
        }
    }
    if let Some(section) = current {
        sections.push(section);
    }
    if sections.is_empty() {
        return Err("model-info is empty".to_owned());
    }
    Ok(sections)
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

    const CONSTANT: &str = "\n[mask]\nid=mask\ntype=bool\ndims=1,1\ndir=input\nconstant=true\n";

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
    fn parses_constant_inputs_beside_the_image() {
        let info = ModelInfo::parse(&(VALID.to_owned() + CONSTANT)).expect("constant parses");
        assert_eq!(info.input().id, "image");
        let [constant] = info.constants() else {
            panic!("one constant expected");
        };
        assert_eq!(constant.description.name, "mask");
        assert_eq!(constant.description.data_type, ScalarType::Bool);
        assert_eq!(constant.value, ConstantValue::Bool(true));
        for (data_type, value, expected) in [
            ("float32", "1.5", ConstantValue::Float32(1.5)),
            ("int64", "-3", ConstantValue::Int64(-3)),
            ("uint8", "7", ConstantValue::Uint8(7)),
            ("bool", "0", ConstantValue::Bool(false)),
        ] {
            let text = VALID.to_owned()
                + &CONSTANT
                    .replace("type=bool", &format!("type={data_type}"))
                    .replace("constant=true", &format!("constant={value}"));
            let info = ModelInfo::parse(&text).expect("typed constant parses");
            assert_eq!(info.constants()[0].value, expected);
        }
    }

    #[test]
    fn rejects_invalid_constants() {
        for invalid in [
            VALID.to_owned() + &CONSTANT.replace("constant=true", "constant=maybe"),
            VALID.to_owned() + &CONSTANT.replace("type=bool", "type=float16"),
            VALID.to_owned() + &CONSTANT.replace("constant=true", "constant=true\nranges=0,1"),
            VALID.to_owned() + &CONSTANT.replace("[mask]", "[input]"),
            VALID.replace(
                "dir=output\n\n[second]",
                "dir=output\nconstant=1\n\n[second]",
            ),
            VALID.replace("ranges=0,255;0,255;0,255", "constant=1"),
        ] {
            assert!(
                ModelInfo::parse(&invalid).is_err(),
                "accepted invalid constant model-info:\n{invalid}"
            );
        }
    }

    #[test]
    fn declared_dimensions_bind_dynamic_model_dimensions() {
        assert!(dims_match(&[None, None, Some(3)], &[1, 400, 3]));
        assert!(!dims_match(&[None, Some(768)], &[1, 400]));
        assert!(!dims_match(&[None, None], &[1, 2, 3]));
    }

    #[test]
    fn requires_exactly_one_image_input() {
        let two_images = VALID.to_owned()
            + "\n[other]\nid=other\ntype=float32\ndims=1,3,2,2\ndir=input\nranges=0,1\n";
        let error = ModelInfo::parse(&two_images).expect_err("two image inputs are rejected");
        assert!(error.contains("exactly one non-constant"), "{error}");
    }

    #[test]
    fn rejects_malformed_essential_fields() {
        for invalid in [
            VALID.replace("version=1.0", "version=2.0"),
            VALID.replace("group-id=test-group", "group-id="),
            VALID.replace("dir=input", "dir=sideways"),
            VALID.replace("dims=1,18", "dims=1,-1"),
            VALID.replace("id=second-output", "id=first-output"),
            VALID.replace("ranges=0,255;0,255;0,255", "ranges=0,1;0,1"),
        ] {
            if ModelInfo::parse(&invalid).is_ok() {
                assert!(invalid.is_empty(), "parser accepted malformed model-info");
            }
        }
    }
}
