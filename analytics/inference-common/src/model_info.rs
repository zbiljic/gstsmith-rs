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
                "unsupported scalar type {value:?}; supported output types are float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, and uint64"
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
        let mut tensors = Vec::new();
        for section in sections.iter().skip(1) {
            let direction = match require(section, "dir", &section.name)? {
                "input" => Direction::Input,
                "output" => Direction::Output,
                value => {
                    return Err(format!(
                        "tensor {:?} has invalid dir {value:?}",
                        section.name
                    ));
                }
            };
            let id = non_empty(require(section, "id", &section.name)?, "id")?.to_owned();
            if !ids.insert(id.clone()) {
                return Err(format!("duplicate tensor id {id:?}"));
            }
            let dims = parse_dims(require(section, "dims", &section.name)?)?;
            if dims.first() != Some(&1) {
                return Err(format!(
                    "tensor {:?} must have a static batch dimension of one",
                    section.name
                ));
            }
            let ranges = if direction == Direction::Input {
                parse_ranges(require(section, "ranges", &section.name)?)?
            } else {
                Vec::new()
            };
            let data_type = ScalarType::parse(require(section, "type", &section.name)?)?;
            if direction == Direction::Input && !data_type.is_supported_input() {
                return Err(format!(
                    "input tensor {:?} type {} is unsupported; inputs must be float32 or uint8",
                    section.name,
                    data_type.as_caps_name()
                ));
            }
            tensors.push((
                direction,
                TensorDescription {
                    name: section.name.clone(),
                    id,
                    data_type,
                    dims,
                    dim_order: DimOrder::parse(
                        section.values.get("dims-order").map(String::as_str),
                    )?,
                    ranges,
                },
            ));
        }
        let inputs: Vec<_> = tensors
            .iter()
            .filter(|(direction, _)| *direction == Direction::Input)
            .map(|(_, tensor)| tensor.clone())
            .collect();
        if inputs.len() != 1 {
            return Err(format!(
                "exactly one input is required, found {}",
                inputs.len()
            ));
        }
        let input = inputs
            .into_iter()
            .next()
            .ok_or_else(|| "missing input".to_owned())?;
        validate_image_input(&input)?;
        let outputs: Vec<_> = tensors
            .into_iter()
            .filter(|(direction, _)| *direction == Direction::Output)
            .map(|(_, tensor)| tensor)
            .collect();
        if outputs.is_empty() {
            return Err("at least one output is required".to_owned());
        }
        Ok(Self {
            group_id,
            input,
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
    #[must_use]
    pub fn outputs(&self) -> &[TensorDescription] {
        &self.outputs
    }

    pub fn image_dimensions(&self) -> Result<(usize, usize), String> {
        match self.input.dims.as_slice() {
            [_, 3, height, width] | [_, height, width, 3] => Ok((*width, *height)),
            _ => Err("input dimensions do not describe a static RGB image".to_owned()),
        }
    }
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

fn validate_image_input(input: &TensorDescription) -> Result<(), String> {
    if input.dims.len() != 4 {
        return Err("the input tensor must have four dimensions".to_owned());
    }
    let channels_first = input.dims.get(1) == Some(&3);
    let channels_last = input.dims.get(3) == Some(&3);
    if channels_first == channels_last {
        return Err("input dimensions must unambiguously describe three image channels".to_owned());
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
