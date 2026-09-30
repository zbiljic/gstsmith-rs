use std::path::Path;
use std::sync::Mutex;

use gst_inference_common::engine::{Engine, InputTensor, OwnedTensor};
use gst_inference_common::model_info::{
    ConstantInput, ConstantValue, ModelInfo, ScalarType, TensorDescription, dims_match,
};
#[cfg(feature = "coreml")]
use ort::ep::ExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};

/// The provider selected for an ORT session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    Cpu,
    #[cfg(feature = "coreml")]
    Coreml,
}

impl Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "coreml")]
            Self::Coreml => "coreml",
        }
    }
}

/// Configuration applied while constructing one ORT session.
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    pub provider: Provider,
    pub intra_threads: Option<usize>,
    pub optimization: GraphOptimizationLevel,
    pub strict_execution_provider: bool,
}

impl EngineOptions {
    pub fn validate(self) -> Result<Self, String> {
        if self.strict_execution_provider && self.provider == Provider::Cpu {
            return Err(format!(
                "strict-execution-provider=true is invalid with execution-provider={}",
                self.provider.name()
            ));
        }
        Ok(self)
    }
}

/// ORT's session API requires mutable access to run. The mutex serializes runs
/// on one session and, importantly, owns all output copies before unlocking.
pub struct OrtEngine {
    session: Mutex<Session>,
    input: TensorDescription,
    constants: Vec<ConstantInput>,
    outputs: Vec<TensorDescription>,
    /// Computes only the outputs model-info declares.
    run_options: RunOptions<HasSelectedOutputs>,
}

impl OrtEngine {
    pub fn load(
        model_file: &Path,
        info: &ModelInfo,
        options: EngineOptions,
    ) -> Result<Self, String> {
        let options = options.validate()?;
        let mut builder = Session::builder()
            .map_err(|error| format!("failed to create ONNX Runtime session builder: {error}"))?
            .with_optimization_level(options.optimization)
            .map_err(|error| format!("failed to configure graph optimization: {error}"))?;
        if let Some(threads) = options.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| format!("failed to configure intra-op threads: {error}"))?;
        }
        match options.provider {
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
        if options.strict_execution_provider {
            builder = builder
                .with_disable_cpu_fallback()
                .map_err(|error| format!("failed to disable CPU execution fallback: {error}"))?;
        }
        let session = builder
            .commit_from_file(model_file)
            .map_err(|error| format!("failed to load ONNX model: {error}"))?;
        validate_session(&session, info)?;
        let selector = info
            .outputs()
            .iter()
            .fold(OutputSelector::no_default(), |selector, output| {
                selector.with(output.name.as_str())
            });
        let run_options = RunOptions::new()
            .map_err(|error| format!("failed to create ONNX Runtime run options: {error}"))?
            .with_outputs(selector);
        Ok(Self {
            session: Mutex::new(session),
            input: info.input().clone(),
            constants: info.constants().to_vec(),
            outputs: info.outputs().to_vec(),
            run_options,
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
        let mut inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(1 + self.constants.len());
        inputs.push((self.input.name.as_str().into(), input_value.into()));
        for constant in &self.constants {
            inputs.push((
                constant.description.name.as_str().into(),
                constant_tensor(constant)?.into(),
            ));
        }
        // ORT's borrowed TensorRef cannot outlive the input buffer while the
        // session runs. Tensor::from_array consumes the input Vec into an
        // owned Value (rather than copying it), while output bytes are copied
        // below so they remain independent of ORT's session allocator.
        let mut session = self
            .session
            .lock()
            .map_err(|_error| "ONNX Runtime session lock is poisoned".to_owned())?;
        let values = session
            .run_with_options(inputs, &self.run_options)
            .map_err(|error| format!("ONNX Runtime inference failed: {error}"))?;
        self.outputs
            .iter()
            .enumerate()
            .map(|(index, description)| {
                let value = values.get(description.name.as_str()).ok_or_else(|| {
                    format!("ONNX Runtime did not return output {:?}", description.name)
                })?;
                let bytes = tensor_bytes(value, description.data_type)
                    .map_err(|error| format!("failed to serialize output {index}: {error}"))?;
                Ok(OwnedTensor {
                    description: description.clone(),
                    bytes,
                })
            })
            .collect()
    }
}

/// A constant input filled with its declared value.
fn constant_tensor(constant: &ConstantInput) -> Result<ort::value::DynValue, String> {
    let dims = constant.description.dims.clone();
    let count = dims.iter().product::<usize>();
    let value = match constant.value {
        ConstantValue::Bool(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
        ConstantValue::Float32(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
        ConstantValue::Float64(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
        ConstantValue::Int32(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
        ConstantValue::Int64(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
        ConstantValue::Uint8(value) => {
            Tensor::from_array((dims, vec![value; count])).map(ort::value::Value::into_dyn)
        }
    };
    value.map_err(|error| {
        format!(
            "failed to construct constant input {:?}: {error}",
            constant.description.name
        )
    })
}

/// Every model input must be declared (the image or a constant) and every
/// declared tensor must exist; outputs may be a declared subset.
fn validate_session(session: &Session, info: &ModelInfo) -> Result<(), String> {
    let declared_inputs = std::iter::once(info.input())
        .chain(
            info.constants()
                .iter()
                .map(|constant| &constant.description),
        )
        .collect::<Vec<_>>();
    for (index, input) in session.inputs().iter().enumerate() {
        let description = declared_inputs
            .iter()
            .find(|description| description.name == input.name())
            .ok_or_else(|| {
                format!(
                    "model input {index} {:?} is not declared in model-info",
                    input.name()
                )
            })?;
        validate_type(input.dtype(), description, "input", index)?;
    }
    for description in &declared_inputs {
        if !session
            .inputs()
            .iter()
            .any(|input| input.name() == description.name)
        {
            return Err(format!(
                "model-info input {:?} is not a model input",
                description.name
            ));
        }
    }
    for (index, description) in info.outputs().iter().enumerate() {
        let output = session
            .outputs()
            .iter()
            .find(|output| output.name() == description.name)
            .ok_or_else(|| {
                format!(
                    "model-info output {:?} is not a model output",
                    description.name
                )
            })?;
        validate_type(output.dtype(), description, "output", index)?;
    }
    Ok(())
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
    // ONNX Runtime reports dynamic dimensions as -1.
    let actual = shape
        .iter()
        .map(|dim| usize::try_from(*dim).ok())
        .collect::<Vec<_>>();
    if !dims_match(&actual, &description.dims) {
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
        ScalarType::Bool => TensorElementType::Bool,
    }
}

fn tensor_bytes(value: &ort::value::DynValue, data_type: ScalarType) -> Result<Vec<u8>, String> {
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
        ScalarType::Bool => value
            .try_extract_tensor::<bool>()
            .map_err(|error| format!("failed to read bool output: {error}"))
            .map(|(_, values)| values.iter().map(|value| u8::from(*value)).collect()),
    }
}
