use std::path::Path;
use std::sync::Mutex;

use gst_inference_common::engine::{Engine, InputTensor, OwnedTensor};
use gst_inference_common::model_info::{ModelInfo, ScalarType, TensorDescription};
#[cfg(feature = "coreml")]
use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, TensorElementType, ValueType};

/// The provider selected for an ORT session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    Cpu,
    #[cfg(feature = "coreml")]
    Coreml,
}

/// ORT's session API requires mutable access to run. The mutex serializes runs
/// on one session and, importantly, owns all output copies before unlocking.
pub struct OrtEngine {
    session: Mutex<Session>,
    input: TensorDescription,
    outputs: Vec<TensorDescription>,
}

impl OrtEngine {
    pub fn load(
        model_file: &Path,
        info: &ModelInfo,
        provider: Provider,
        intra_threads: Option<usize>,
        optimization: GraphOptimizationLevel,
    ) -> Result<Self, String> {
        let mut builder = Session::builder()
            .map_err(|error| format!("failed to create ONNX Runtime session builder: {error}"))?
            .with_optimization_level(optimization)
            .map_err(|error| format!("failed to configure graph optimization: {error}"))?;
        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| format!("failed to configure intra-op threads: {error}"))?;
        }
        match provider {
            Provider::Cpu => {
                builder = builder
                    .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
                    .map_err(|error| {
                        format!("failed to configure CPU execution provider: {error}")
                    })?;
            }
            #[cfg(feature = "coreml")]
            Provider::Coreml => {
                let coreml = ort::ep::CoreML::default();
                let available = coreml.is_available().map_err(|error| {
                    format!("failed to query CoreML execution provider: {error}")
                })?;
                if !available {
                    return Err(
                        "CoreML execution provider is unavailable in this ONNX Runtime build"
                            .to_owned(),
                    );
                }
                builder = builder
                    .with_execution_providers([coreml.build().error_on_failure()])
                    .map_err(|error| {
                        format!("failed to configure CoreML execution provider: {error}")
                    })?;
            }
        }
        let session = builder
            .commit_from_file(model_file)
            .map_err(|error| format!("failed to load ONNX model: {error}"))?;
        validate_session(&session, info)?;
        Ok(Self {
            session: Mutex::new(session),
            input: info.input().clone(),
            outputs: info.outputs().to_vec(),
        })
    }
}

impl Engine for OrtEngine {
    fn run(&self, input: InputTensor) -> Result<Vec<OwnedTensor>, String> {
        let input_value = match (self.input.data_type, input) {
            (ScalarType::Float32, InputTensor::Float32(values)) => {
                Tensor::from_array((self.input.dims.clone(), values))
                    .map(ort::value::Value::into_dyn)
            }
            (ScalarType::Uint8, InputTensor::Uint8(values)) => {
                Tensor::from_array((self.input.dims.clone(), values))
                    .map(ort::value::Value::into_dyn)
            }
            _ => return Err("preprocessor produced the wrong input scalar type".to_owned()),
        }
        .map_err(|error| format!("failed to construct ORT input tensor: {error}"))?;
        // ORT's borrowed TensorRef cannot outlive the input buffer while the
        // session runs. Tensor::from_array consumes the input Vec into an
        // owned Value (rather than copying it), while output bytes are copied
        // below so they remain independent of ORT's session allocator.
        let mut session = self
            .session
            .lock()
            .map_err(|_error| "ONNX Runtime session lock is poisoned".to_owned())?;
        let values = session
            .run(ort::inputs![input_value])
            .map_err(|error| format!("ONNX Runtime inference failed: {error}"))?;
        if values.len() != self.outputs.len() {
            return Err("ONNX Runtime returned an unexpected number of outputs".to_owned());
        }
        values
            .iter()
            .zip(&self.outputs)
            .enumerate()
            .map(|(index, ((_, value), description))| {
                let bytes = tensor_bytes(&value, description.data_type)
                    .map_err(|error| format!("failed to serialize output {index}: {error}"))?;
                Ok(OwnedTensor {
                    description: description.clone(),
                    bytes,
                })
            })
            .collect()
    }
}

fn validate_session(session: &Session, info: &ModelInfo) -> Result<(), String> {
    if session.inputs().len() != 1 {
        return Err(format!(
            "model has {} inputs; exactly one is supported",
            session.inputs().len()
        ));
    }
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| "model did not expose its input".to_owned())?;
    validate_name(input.name(), &info.input().name, "input", 0)?;
    validate_type(input.dtype(), info.input(), "input", 0)?;
    if session.outputs().len() != info.outputs().len() {
        return Err(format!(
            "model has {} outputs but model-info declares {}",
            session.outputs().len(),
            info.outputs().len()
        ));
    }
    for (index, (output, description)) in session.outputs().iter().zip(info.outputs()).enumerate() {
        validate_name(output.name(), &description.name, "output", index)?;
        validate_type(output.dtype(), description, "output", index)?;
    }
    Ok(())
}

fn validate_name(
    discovered: &str,
    expected: &str,
    direction: &str,
    index: usize,
) -> Result<(), String> {
    if discovered == expected {
        Ok(())
    } else {
        Err(format!(
            "{direction} {index} name mismatch: model {discovered:?}, model-info {expected:?}"
        ))
    }
}

fn validate_type(
    value: &ValueType,
    description: &TensorDescription,
    direction: &str,
    index: usize,
) -> Result<(), String> {
    let ValueType::Tensor { ty, shape, .. } = value else {
        return Err(format!("{direction} {index} is not a tensor"));
    };
    let expected_type = tensor_element_type(description.data_type);
    if *ty != expected_type {
        return Err(format!(
            "{direction} {index} scalar type mismatch: model {ty:?}, model-info {expected_type:?}"
        ));
    }
    let actual = shape
        .iter()
        .copied()
        .map(|dim| usize::try_from(dim).ok())
        .collect::<Option<Vec<_>>>();
    if actual.as_deref() != Some(description.dims.as_slice()) {
        return Err(format!(
            "{direction} {index} dimensions mismatch: model {shape:?}, model-info {:?}",
            description.dims
        ));
    }
    Ok(())
}

fn tensor_element_type(data_type: ScalarType) -> TensorElementType {
    match data_type {
        ScalarType::Float16 => TensorElementType::Float16,
        ScalarType::Float64 => TensorElementType::Float64,
        ScalarType::Float32 => TensorElementType::Float32,
        ScalarType::Int8 => TensorElementType::Int8,
        ScalarType::Int16 => TensorElementType::Int16,
        ScalarType::Int32 => TensorElementType::Int32,
        ScalarType::Int64 => TensorElementType::Int64,
        ScalarType::Uint8 => TensorElementType::Uint8,
        ScalarType::Uint16 => TensorElementType::Uint16,
        ScalarType::Uint32 => TensorElementType::Uint32,
        ScalarType::Uint64 => TensorElementType::Uint64,
    }
}

fn tensor_bytes(
    value: &ort::value::ValueRef<'_, ort::value::DynValueTypeMarker>,
    data_type: ScalarType,
) -> Result<Vec<u8>, String> {
    macro_rules! scalar_bytes {
        ($type:ty, $label:literal) => {
            value
                .try_extract_tensor::<$type>()
                .map_err(|error| format!("failed to read {} output: {error}", $label))
                .map(|(_, values)| {
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect()
                })
        };
    }
    match data_type {
        ScalarType::Float16 => value
            .try_extract_tensor::<half::f16>()
            .map_err(|error| format!("failed to read float16 output: {error}"))
            .map(|(_, values)| {
                values
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect()
            }),
        ScalarType::Float64 => scalar_bytes!(f64, "float64"),
        ScalarType::Float32 => scalar_bytes!(f32, "float32"),
        ScalarType::Int8 => scalar_bytes!(i8, "int8"),
        ScalarType::Int16 => scalar_bytes!(i16, "int16"),
        ScalarType::Int32 => scalar_bytes!(i32, "int32"),
        ScalarType::Int64 => scalar_bytes!(i64, "int64"),
        ScalarType::Uint8 => value
            .try_extract_tensor::<u8>()
            .map_err(|error| format!("failed to read uint8 output: {error}"))
            .map(|(_, values)| values.to_vec()),
        ScalarType::Uint16 => scalar_bytes!(u16, "uint16"),
        ScalarType::Uint32 => scalar_bytes!(u32, "uint32"),
        ScalarType::Uint64 => scalar_bytes!(u64, "uint64"),
    }
}
