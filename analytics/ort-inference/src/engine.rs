use std::path::Path;
use std::sync::Mutex;

use gst_inference_common::engine::{Engine, InputTensor, OwnedTensor, TensorEngine, TensorValues};
use gst_inference_common::model_info::{
    ModelInfo, ScalarType, TensorDescription, TensorModelInfo, dims_match,
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
#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub provider: Provider,
    pub intra_threads: Option<usize>,
    pub optimization: GraphOptimizationLevel,
    pub strict_execution_provider: bool,
    #[cfg(feature = "coreml")]
    pub coreml: crate::coreml::CoreMlOptions,
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
struct SessionPlan {
    session: Mutex<Session>,
    /// Output constraints combine model-info with fixed ONNX axes.
    outputs: Vec<TensorDescription>,
    /// Requests only the outputs model-info declares.
    run_options: RunOptions<HasSelectedOutputs>,
}

pub struct OrtEngine {
    plan: SessionPlan,
    input: TensorDescription,
}

pub struct OrtTensorEngine {
    plan: SessionPlan,
    inputs: Vec<TensorDescription>,
}

impl OrtEngine {
    pub fn load(
        model_file: &Path,
        info: &ModelInfo,
        options: EngineOptions,
    ) -> Result<Self, String> {
        let input = info.input().clone();
        let plan = SessionPlan::load(
            model_file,
            std::slice::from_ref(&input),
            info.outputs(),
            options,
        )?;
        Ok(Self { plan, input })
    }
}

impl OrtTensorEngine {
    pub fn load(
        model_file: &Path,
        info: &TensorModelInfo,
        options: EngineOptions,
    ) -> Result<Self, String> {
        let plan = SessionPlan::load(model_file, info.inputs(), info.outputs(), options)?;
        Ok(Self {
            plan,
            inputs: info.inputs().to_vec(),
        })
    }
}

impl SessionPlan {
    fn load(
        model_file: &Path,
        provided: &[TensorDescription],
        outputs: &[TensorDescription],
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
                let coreml = options.coreml.execution_provider();
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
        let outputs = validate_session(&session, provided, outputs)?;
        let selector = outputs
            .iter()
            .fold(OutputSelector::no_default(), |selector, output| {
                selector.with(output.name.as_str())
            });
        let run_options = RunOptions::new()
            .map_err(|error| format!("failed to create ONNX Runtime run options: {error}"))?
            .with_outputs(selector);
        Ok(Self {
            session: Mutex::new(session),
            outputs,
            run_options,
        })
    }

    /// Run with the provided inputs (name, value).
    fn run(&self, provided: Vec<(&str, ort::value::DynValue)>) -> Result<Vec<OwnedTensor>, String> {
        // ORT's borrowed TensorRef cannot outlive the input buffer while the
        // session runs. Tensor::from_array consumes the input Vec into an
        // owned Value (rather than copying it), while output bytes are copied
        // below so they remain independent of ORT's session allocator.
        let mut session = self
            .session
            .lock()
            .map_err(|_error| "ONNX Runtime session lock is poisoned".to_owned())?;
        // A repeated ONNX dimension symbol describes the same size across inputs.
        // ORT may otherwise accept mismatches through operator broadcasting.
        let mut symbols = std::collections::BTreeMap::new();
        for (name, value) in &provided {
            let model = session
                .inputs()
                .iter()
                .find(|input| input.name() == *name)
                .ok_or_else(|| format!("unknown model input {name:?}"))?;
            validate_input_shape(name, model.dtype(), value.dtype(), &mut symbols)?;
        }
        let mut inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(provided.len());
        for (name, value) in provided {
            inputs.push((name.into(), value.into()));
        }
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
                validate_type(value.dtype(), description, "output", index)?;
                let ValueType::Tensor { shape, .. } = value.dtype() else {
                    return Err(format!("output {index} is not a tensor"));
                };
                let dims = shape
                    .iter()
                    .map(|dim| usize::try_from(*dim))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_error| format!("output {index} has unresolved dimensions"))?;
                let size = description.validate_dims(&dims)?;
                let bytes = tensor_bytes(value, description.data_type)
                    .map_err(|error| format!("failed to serialize output {index}: {error}"))?;
                if bytes.len() != size {
                    return Err(format!(
                        "output {index} byte size mismatch: expected {size}, got {}",
                        bytes.len()
                    ));
                }
                Ok(OwnedTensor {
                    dims,
                    description: description.clone(),
                    bytes,
                })
            })
            .collect()
    }
}

impl Engine for OrtEngine {
    fn run(&self, input: InputTensor) -> Result<Vec<OwnedTensor>, String> {
        let input_value = match (self.input.data_type, input) {
            (ScalarType::Float32, InputTensor::Float32(values)) => {
                Tensor::from_array((self.input.concrete_dims()?, values))
                    .map(ort::value::Value::into_dyn)
            }
            (ScalarType::Uint8, InputTensor::Uint8(values)) => {
                Tensor::from_array((self.input.concrete_dims()?, values))
                    .map(ort::value::Value::into_dyn)
            }
            _ => return Err("preprocessor produced the wrong input scalar type".to_owned()),
        }
        .map_err(|error| format!("failed to construct ORT input tensor: {error}"))?;
        self.plan.run(vec![(self.input.name.as_str(), input_value)])
    }
}

impl TensorEngine for OrtTensorEngine {
    fn run(&self, inputs: &[OwnedTensor]) -> Result<Vec<OwnedTensor>, String> {
        if inputs.len() != self.inputs.len() {
            return Err("wrong number of input tensors".to_owned());
        }
        let provided = inputs
            .iter()
            .zip(&self.inputs)
            .map(|(input, description)| {
                description.validate_dims(&input.dims)?;
                owned_tensor_value(input).map(|value| (description.name.as_str(), value))
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.plan.run(provided)
    }
}

fn validate_input_shape(
    name: &str,
    model: &ValueType,
    actual: &ValueType,
    symbols: &mut std::collections::BTreeMap<String, i64>,
) -> Result<(), String> {
    let (
        ValueType::Tensor {
            shape: expected,
            dimension_symbols,
            ..
        },
        ValueType::Tensor { shape: actual, .. },
    ) = (model, actual)
    else {
        return Err(format!("input {name:?} must be a tensor"));
    };
    if expected.len() != actual.len() {
        return Err(format!("input {name:?} rank mismatch"));
    }
    for ((expected, actual), symbol) in expected
        .iter()
        .zip(actual.iter())
        .zip(dimension_symbols.iter())
    {
        if *expected != -1 && expected != actual {
            return Err(format!(
                "input {name:?} dimensions mismatch: model axis {expected}, tensor axis {actual}"
            ));
        }
        if !symbol.is_empty()
            && let Some(previous) = symbols.insert(symbol.clone(), *actual)
            && previous != *actual
        {
            return Err(format!(
                "input {name:?} dimension symbol {symbol:?} mismatch: {previous} versus {actual}"
            ));
        }
    }
    Ok(())
}

/// An ORT tensor with the owned tensor's type, shape, and values.
fn owned_tensor_value(input: &OwnedTensor) -> Result<ort::value::DynValue, String> {
    let dims = input.dims.clone();
    let value = match TensorValues::decode(input)? {
        TensorValues::Float16(bits) => Tensor::from_array((
            dims,
            bits.into_iter()
                .map(half::f16::from_bits)
                .collect::<Vec<_>>(),
        ))
        .map(ort::value::Value::into_dyn),
        TensorValues::Float32(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Float64(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Int8(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Int16(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Int32(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Int64(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Uint8(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Uint16(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Uint32(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
        TensorValues::Uint64(values) => {
            Tensor::from_array((dims, values)).map(ort::value::Value::into_dyn)
        }
    };
    value.map_err(|error| {
        format!(
            "failed to construct ORT input tensor {}: {error}",
            input.description.id
        )
    })
}

/// Validate the declarations and retain the intersection of output constraints.
/// Every model input and declared tensor must exist; outputs may be a subset.
fn validate_session(
    session: &Session,
    provided: &[TensorDescription],
    outputs: &[TensorDescription],
) -> Result<Vec<TensorDescription>, String> {
    for (index, input) in session.inputs().iter().enumerate() {
        let description = provided
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
    for description in provided {
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
    let mut constrained_outputs = outputs.to_vec();
    for (index, description) in constrained_outputs.iter_mut().enumerate() {
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
        let ValueType::Tensor { shape, .. } = output.dtype() else {
            return Err(format!("output {index} is not a tensor"));
        };
        // A wildcard relaxes model-info only, never a fixed axis in the model.
        // Keep this engine constraint separate from the declared public caps.
        for (declared, model) in description.dims.iter_mut().zip(shape.iter()) {
            if *declared == -1 {
                *declared = i32::try_from(*model)
                    .map_err(|_error| format!("output {index} dimension exceeds tensor limits"))?;
            }
        }
    }
    Ok(constrained_outputs)
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
    if shape.is_empty()
        || shape
            .iter()
            .any(|dim| *dim != -1 && (*dim <= 0 || *dim > i64::from(i32::MAX)))
    {
        return Err(format!(
            "{direction} {index} has unsupported dimensions {shape:?}"
        ));
    }
    // ONNX Runtime reports dynamic dimensions as -1.
    let actual = shape
        .iter()
        .map(|dim| usize::try_from(*dim).ok())
        .collect::<Vec<_>>();
    if !dims_match(&actual, &description.dims) {
        return Err(format!(
            "{direction} {index} dimensions mismatch: actual {shape:?}, required {:?}",
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_runtime_outputs_that_differ_from_bound_dimensions()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = tempfile::NamedTempFile::new()?;
        std::fs::write(
            model.path(),
            include_bytes!("../../inference-common/tests/fixtures/masked-sequence.onnx"),
        )?;
        let contents =
            include_str!("../../inference-common/tests/fixtures/masked-sequence.onnx.modelinfo")
                .replace(
                    "id=scaled\ntype=float32\ndims=1,3,2",
                    "id=scaled\ntype=float32\ndims=1,4,2",
                );
        let info = TensorModelInfo::parse(&contents).map_err(std::io::Error::other)?;
        let engine = OrtTensorEngine::load(
            model.path(),
            &info,
            EngineOptions {
                provider: Provider::Cpu,
                intra_threads: None,
                optimization: GraphOptimizationLevel::Level3,
                strict_execution_provider: false,
                #[cfg(feature = "coreml")]
                coreml: crate::coreml::CoreMlOptions::default(),
            },
        )
        .map_err(std::io::Error::other)?;
        let inputs = info
            .inputs()
            .iter()
            .map(|description| OwnedTensor {
                description: description.clone(),
                dims: description.concrete_dims().expect("fixed fixture"),
                bytes: vec![
                    0;
                    description
                        .concrete_dims()
                        .expect("fixed fixture")
                        .iter()
                        .product::<usize>()
                        * description.data_type.size()
                ],
            })
            .collect::<Vec<_>>();
        match engine.run(&inputs) {
            Err(error) if error.contains("output 0 dimensions mismatch") => Ok(()),
            result => Err(std::io::Error::other(format!(
                "expected output shape error, got {result:?}"
            ))
            .into()),
        }
    }
}
