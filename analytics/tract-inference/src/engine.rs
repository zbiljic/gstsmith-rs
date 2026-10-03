pub use gst_inference_common::engine::{Engine, TensorEngine};
#[cfg(feature = "tract")]
pub use gst_inference_common::engine::{InputTensor, OwnedTensor, TensorValues};

#[cfg(feature = "tract")]
pub mod tract {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::infer::Factoid;

    use super::{Engine, InputTensor, OwnedTensor, TensorEngine, TensorValues};
    use crate::tractinference::imp::ExecutionProvider;
    use gst_inference_common::model_info::{
        ModelInfo, ScalarType, TensorDescription, TensorModelInfo, dims_match,
    };

    /// A runnable model whose inputs are provided tensors.
    struct Plan {
        runnable: std::sync::Arc<TypedRunnableModel>,
        slots: Vec<usize>,
        outputs: Vec<TensorDescription>,
    }

    pub struct TractEngine {
        plan: Plan,
        input: TensorDescription,
    }

    pub struct TractTensorEngine {
        plan: Plan,
    }

    impl TractEngine {
        pub fn load(
            model_file: &std::path::Path,
            info: &ModelInfo,
            execution_provider: ExecutionProvider,
        ) -> Result<Self, String> {
            let input = info.input().clone();
            let plan = Plan::load(
                model_file,
                std::slice::from_ref(&input),
                info.outputs(),
                execution_provider,
            )?;
            Ok(Self { plan, input })
        }
    }

    impl TractTensorEngine {
        pub fn load(
            model_file: &std::path::Path,
            info: &TensorModelInfo,
            execution_provider: ExecutionProvider,
        ) -> Result<Self, String> {
            let plan = Plan::load(
                model_file,
                info.inputs(),
                info.outputs(),
                execution_provider,
            )?;
            Ok(Self { plan })
        }
    }

    impl Plan {
        fn load(
            model_file: &std::path::Path,
            provided: &[TensorDescription],
            outputs: &[TensorDescription],
            execution_provider: ExecutionProvider,
        ) -> Result<Self, String> {
            let mut model = tract_onnx::onnx()
                .model_for_path(model_file)
                .map_err(|error| format!("failed to load ONNX model: {error}"))?;
            let (slots, facts) = input_slots(&model, provided)?;
            for (index, fact) in facts.into_iter().enumerate() {
                model = model.with_input_fact(index, fact).map_err(|error| {
                    format!("failed to specialize model input {index}: {error}")
                })?;
            }
            let selected = output_outlets(&model, outputs)?;
            let model = model
                .with_output_outlets(&selected)
                .map_err(|error| format!("failed to select model outputs: {error}"))?;
            let mut model = model
                .into_typed()
                .map_err(|error| format!("failed to convert model to a typed graph: {error}"))?;
            apply_execution_provider(&mut model, execution_provider)?;
            let model = model
                .into_optimized()
                .map_err(|error| format!("failed to optimize model: {error}"))?;
            let runtime_outputs = model
                .output_outlets()
                .map_err(|error| format!("failed to inspect model outputs: {error}"))?;
            if runtime_outputs.len() != outputs.len() {
                return Err(format!(
                    "model has {} outputs but model-info declares {}",
                    runtime_outputs.len(),
                    outputs.len()
                ));
            }
            for (index, descriptor) in outputs.iter().enumerate() {
                let fact = model
                    .output_fact(index)
                    .map_err(|error| format!("failed to inspect output {index}: {error}"))?;
                validate_typed_fact(fact, descriptor, "output", index)?;
            }
            let runnable = model
                .into_runnable()
                .map_err(|error| format!("failed to create runnable model: {error}"))?;
            Ok(Self {
                runnable,
                slots,
                outputs: outputs.to_vec(),
            })
        }

        /// Run with `provided` in the order the plan was loaded with.
        fn run(&self, provided: Vec<TValue>) -> Result<Vec<OwnedTensor>, String> {
            let mut provided = provided.into_iter().map(Some).collect::<Vec<_>>();
            let inputs = self
                .slots
                .iter()
                .map(|index| {
                    provided
                        .get_mut(*index)
                        .and_then(Option::take)
                        .ok_or_else(|| format!("model input {index} was not provided"))
                })
                .collect::<Result<TVec<_>, String>>()?;
            let runtime_outputs = self
                .runnable
                .run(inputs)
                .map_err(|error| format!("Tract execution failed: {error}"))?;
            if runtime_outputs.len() != self.outputs.len() {
                return Err("Tract returned an unexpected number of outputs".to_owned());
            }
            runtime_outputs
                .into_iter()
                .zip(&self.outputs)
                .map(|(value, description)| {
                    let tensor = value.into_tensor();
                    let bytes = tensor_bytes(&tensor, description.data_type)?;
                    Ok(OwnedTensor {
                        description: description.clone(),
                        bytes,
                    })
                })
                .collect()
        }
    }

    /// Map every model input to a provided input,
    /// validate it, and derive the static fact model-info binds it to.
    fn input_slots(
        model: &InferenceModel,
        provided: &[TensorDescription],
    ) -> Result<(Vec<usize>, Vec<InferenceFact>), String> {
        let outlets = model
            .input_outlets()
            .map_err(|error| format!("failed to inspect model inputs: {error}"))?;
        let mut slots = Vec::with_capacity(outlets.len());
        let mut facts = Vec::with_capacity(outlets.len());
        let mut names = Vec::with_capacity(outlets.len());
        for (index, outlet) in outlets.iter().enumerate() {
            let name = outlet_name(model, *outlet).unwrap_or_default();
            let (slot, description) = provided
                .iter()
                .enumerate()
                .find(|(_, description)| description.name == name)
                .ok_or_else(|| {
                    format!("model input {index} {name:?} is not declared in model-info")
                })?;
            let runtime_fact = model
                .input_fact(index)
                .map_err(|error| format!("failed to inspect model input {index}: {error}"))?;
            validate_fact(runtime_fact, description, "input", index)?;
            facts.push(InferenceFact::dt_shape(
                datum_type(description.data_type),
                description.dims.clone(),
            ));
            slots.push(slot);
            names.push(name.to_owned());
        }
        for description in provided {
            if !names.contains(&description.name) {
                return Err(format!(
                    "model-info input {:?} is not a model input",
                    description.name
                ));
            }
        }
        Ok((slots, facts))
    }

    /// The model outputs model-info declares, in model-info order.
    fn output_outlets(
        model: &InferenceModel,
        outputs: &[TensorDescription],
    ) -> Result<Vec<OutletId>, String> {
        let outlets = model
            .output_outlets()
            .map_err(|error| format!("failed to inspect model outputs: {error}"))?;
        outputs
            .iter()
            .map(|description| {
                outlets
                    .iter()
                    .copied()
                    .find(|outlet| outlet_name(model, *outlet) == Some(description.name.as_str()))
                    .ok_or_else(|| {
                        format!(
                            "model-info output {:?} is not a model output",
                            description.name
                        )
                    })
            })
            .collect()
    }

    fn apply_execution_provider(
        model: &mut TypedModel,
        execution_provider: ExecutionProvider,
    ) -> Result<(), String> {
        match execution_provider {
            ExecutionProvider::Cpu => Ok(()),
            ExecutionProvider::Metal => apply_metal_transform(model),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn apply_metal_transform(model: &mut TypedModel) -> Result<(), String> {
        use tract_metal::MetalTransform;
        use tract_onnx::tract_core::transform::ModelTransform;

        MetalTransform::default()
            .transform(model)
            .map_err(|error| format!("failed to apply the Tract Metal transform: {error}"))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn apply_metal_transform(_model: &mut TypedModel) -> Result<(), String> {
        #[cfg(not(target_os = "macos"))]
        return Err("Metal execution is only supported on macOS".to_owned());
        #[cfg(all(target_os = "macos", not(feature = "metal")))]
        return Err("Metal support was not compiled; rebuild with the `metal` feature".to_owned());
    }

    impl Engine for TractEngine {
        fn run(&self, input: InputTensor) -> Result<Vec<OwnedTensor>, String> {
            let tensor = match (self.input.data_type, input) {
                (ScalarType::Float32, InputTensor::Float32(values)) => {
                    Tensor::from_shape(&self.input.dims, &values)
                        .map_err(|error| format!("failed to make float input tensor: {error}"))?
                }
                (ScalarType::Uint8, InputTensor::Uint8(values)) => {
                    Tensor::from_shape(&self.input.dims, &values)
                        .map_err(|error| format!("failed to make byte input tensor: {error}"))?
                }
                _ => return Err("preprocessor produced the wrong input scalar type".to_owned()),
            };
            self.plan.run(vec![TValue::from(tensor)])
        }
    }

    impl TensorEngine for TractTensorEngine {
        fn run(&self, inputs: &[OwnedTensor]) -> Result<Vec<OwnedTensor>, String> {
            let values = inputs
                .iter()
                .map(|input| tensor_from_owned(input).map(TValue::from))
                .collect::<Result<Vec<_>, String>>()?;
            self.plan.run(values)
        }
    }

    /// A Tract tensor with the owned tensor's type, shape, and values.
    fn tensor_from_owned(input: &OwnedTensor) -> Result<Tensor, String> {
        let dims = &input.description.dims;
        let tensor = match TensorValues::decode(input)? {
            TensorValues::Float16(bits) => Tensor::from_shape(
                dims,
                &bits.into_iter().map(f16::from_bits).collect::<Vec<_>>(),
            ),
            TensorValues::Float32(values) => Tensor::from_shape(dims, &values),
            TensorValues::Float64(values) => Tensor::from_shape(dims, &values),
            TensorValues::Int8(values) => Tensor::from_shape(dims, &values),
            TensorValues::Int16(values) => Tensor::from_shape(dims, &values),
            TensorValues::Int32(values) => Tensor::from_shape(dims, &values),
            TensorValues::Int64(values) => Tensor::from_shape(dims, &values),
            TensorValues::Uint8(values) => Tensor::from_shape(dims, &values),
            TensorValues::Uint16(values) => Tensor::from_shape(dims, &values),
            TensorValues::Uint32(values) => Tensor::from_shape(dims, &values),
            TensorValues::Uint64(values) => Tensor::from_shape(dims, &values),
        };
        tensor.map_err(|error| {
            format!(
                "failed to make input tensor {}: {error}",
                input.description.id
            )
        })
    }

    fn outlet_name(model: &InferenceModel, outlet: OutletId) -> Option<&str> {
        model
            .outlet_label(outlet)
            .or_else(|| Some(model.node(outlet.node).name.as_str()))
    }

    fn validate_fact(
        fact: &InferenceFact,
        descriptor: &TensorDescription,
        direction: &str,
        index: usize,
    ) -> Result<(), String> {
        let expected = datum_type(descriptor.data_type);
        let actual_type = fact
            .datum_type
            .concretize()
            .ok_or_else(|| format!("{direction} {index} has a dynamic scalar type"))?;
        if actual_type != expected {
            return Err(format!(
                "{direction} {index} scalar type mismatch: model {actual_type:?}, model-info {expected:?}"
            ));
        }
        if fact.shape.is_open() {
            return Ok(());
        }
        // Symbolic model dimensions are bound by the model-info dimensions.
        let shape = fact
            .shape
            .dims()
            .map(|dim| {
                dim.concretize()
                    .and_then(|dim| dim.as_i64())
                    .and_then(|dim| usize::try_from(dim).ok())
            })
            .collect::<Vec<_>>();
        if !dims_match(&shape, &descriptor.dims) {
            return Err(format!(
                "{direction} {index} dimensions mismatch: model {:?}, model-info {:?}",
                fact.shape, descriptor.dims
            ));
        }
        Ok(())
    }

    fn validate_typed_fact(
        fact: &TypedFact,
        descriptor: &TensorDescription,
        direction: &str,
        index: usize,
    ) -> Result<(), String> {
        let expected = datum_type(descriptor.data_type);
        if fact.datum_type != expected {
            return Err(format!(
                "{direction} {index} scalar type mismatch: model {:?}, model-info {:?}",
                fact.datum_type, expected
            ));
        }
        let shape = fact
            .shape
            .as_concrete()
            .ok_or_else(|| format!("{direction} {index} has dynamic dimensions"))?;
        let actual = shape.to_vec();
        if actual != descriptor.dims {
            return Err(format!(
                "{direction} {index} dimensions mismatch: model {actual:?}, model-info {:?}",
                descriptor.dims
            ));
        }
        Ok(())
    }

    fn datum_type(data_type: ScalarType) -> DatumType {
        match data_type {
            ScalarType::Float16 => f16::datum_type(),
            ScalarType::Float64 => f64::datum_type(),
            ScalarType::Float32 => f32::datum_type(),
            ScalarType::Int8 => i8::datum_type(),
            ScalarType::Int16 => i16::datum_type(),
            ScalarType::Int32 => i32::datum_type(),
            ScalarType::Int64 => i64::datum_type(),
            ScalarType::Uint8 => u8::datum_type(),
            ScalarType::Uint16 => u16::datum_type(),
            ScalarType::Uint32 => u32::datum_type(),
            ScalarType::Uint64 => u64::datum_type(),
        }
    }

    fn tensor_bytes(tensor: &Tensor, data_type: ScalarType) -> Result<Vec<u8>, String> {
        macro_rules! scalar_bytes {
            ($type:ty, $label:literal) => {
                tensor
                    .to_plain_array_view::<$type>()
                    .map_err(|error| format!("failed to read {} output: {error}", $label))?
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
        }
        match data_type {
            ScalarType::Float16 => tensor
                .to_plain_array_view::<f16>()
                .map_err(|error| format!("failed to read float16 output: {error}"))
                .map(|values| {
                    values
                        .iter()
                        .flat_map(|value| value.to_bits().to_le_bytes())
                        .collect()
                }),
            ScalarType::Float64 => Ok(scalar_bytes!(f64, "float64")),
            ScalarType::Float32 => Ok(scalar_bytes!(f32, "float32")),
            ScalarType::Int8 => Ok(scalar_bytes!(i8, "int8")),
            ScalarType::Int16 => Ok(scalar_bytes!(i16, "int16")),
            ScalarType::Int32 => Ok(scalar_bytes!(i32, "int32")),
            ScalarType::Int64 => Ok(scalar_bytes!(i64, "int64")),
            ScalarType::Uint8 => tensor
                .to_plain_array_view::<u8>()
                .map_err(|error| format!("failed to read uint8 output: {error}"))
                .map(|values| values.iter().copied().collect()),
            ScalarType::Uint16 => Ok(scalar_bytes!(u16, "uint16")),
            ScalarType::Uint32 => Ok(scalar_bytes!(u32, "uint32")),
            ScalarType::Uint64 => Ok(scalar_bytes!(u64, "uint64")),
        }
    }

    #[cfg(test)]
    mod serialization_tests {
        use super::*;

        #[test]
        fn serializes_float16_and_int32_outputs_without_conversion()
        -> Result<(), Box<dyn std::error::Error>> {
            let float_values = [f16::from_f32(1.0), f16::from_f32(-2.0)];
            let float_tensor = Tensor::from_shape(&[2], &float_values)?;
            let float_bytes =
                tensor_bytes(&float_tensor, ScalarType::Float16).map_err(std::io::Error::other)?;
            let expected_float_bytes = float_values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            if float_bytes != expected_float_bytes
                || datum_type(ScalarType::Float16) != f16::datum_type()
            {
                return Err(
                    std::io::Error::other("float16 output bytes or datum type changed").into(),
                );
            }

            let int_values = [1_i32, -2_i32];
            let int_tensor = Tensor::from_shape(&[2], &int_values)?;
            let int_bytes =
                tensor_bytes(&int_tensor, ScalarType::Int32).map_err(std::io::Error::other)?;
            let expected_int_bytes = int_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            if int_bytes != expected_int_bytes || datum_type(ScalarType::Int32) != i32::datum_type()
            {
                return Err(
                    std::io::Error::other("int32 output bytes or datum type changed").into(),
                );
            }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "tract"))]
mod tests {
    use std::io::Write;

    use super::{Engine, InputTensor, tract::TractEngine};
    use gst_inference_common::model_info::ModelInfo;

    #[test]
    fn runs_a_static_two_output_onnx_model() -> Result<(), Box<dyn std::error::Error>> {
        let mut model_file = tempfile::NamedTempFile::new()?;
        model_file.write_all(include_bytes!(
            "../../inference-common/tests/fixtures/identity.onnx"
        ))?;
        let info = ModelInfo::parse(include_str!(
            "../../inference-common/tests/fixtures/identity.onnx.modelinfo"
        ))
        .map_err(std::io::Error::other)?;
        let engine = TractEngine::load(
            model_file.path(),
            &info,
            crate::tractinference::imp::ExecutionProvider::Cpu,
        )
        .map_err(std::io::Error::other)?;
        let outputs = engine
            .run(InputTensor::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .map_err(std::io::Error::other)?;
        if outputs.len() != 2 {
            return Err(std::io::Error::other("Tract did not produce two outputs").into());
        }
        let first = outputs
            .first()
            .ok_or_else(|| std::io::Error::other("Tract omitted the first output"))?;
        let second = outputs
            .get(1)
            .ok_or_else(|| std::io::Error::other("Tract omitted the second output"))?;
        if first.description.id != "first" || second.description.id != "second" {
            return Err(
                std::io::Error::other("Tract output order did not match model-info").into(),
            );
        }
        let values: Vec<_> = first
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        if values != [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            return Err(
                std::io::Error::other("Tract identity output values were incorrect").into(),
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_runtime_model_info_mismatches() -> Result<(), Box<dyn std::error::Error>> {
        let mut model_file = tempfile::NamedTempFile::new()?;
        model_file.write_all(include_bytes!(
            "../../inference-common/tests/fixtures/identity.onnx"
        ))?;
        let fixture = include_str!("../../inference-common/tests/fixtures/identity.onnx.modelinfo");
        for invalid in [
            fixture.replacen("[x]", "[wrong-input]", 1),
            fixture.replacen("type=float32", "type=uint8", 1),
            fixture.replacen("dims=1,1,2,3", "dims=1,1,1,3", 1),
            fixture.replacen("[y]", "[wrong-output]", 1),
            fixture.replacen(
                "[y]\nid=first\ntype=float32",
                "[y]\nid=first\ntype=int32",
                1,
            ),
            fixture.replacen(
                "[y]\nid=first\ntype=float32\ndims=1,1,2,3",
                "[y]\nid=first\ntype=float32\ndims=1,1,3,2",
                1,
            ),
        ] {
            let info = ModelInfo::parse(&invalid).map_err(std::io::Error::other)?;
            if TractEngine::load(
                model_file.path(),
                &info,
                crate::tractinference::imp::ExecutionProvider::Cpu,
            )
            .is_ok()
            {
                return Err(
                    std::io::Error::other("runtime/model-info mismatch was accepted").into(),
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_fixture_translates_a_convolution_to_a_device_operation()
    -> Result<(), Box<dyn std::error::Error>> {
        use tract_onnx::prelude::*;
        use tract_onnx::tract_core::transform::ModelTransform;

        let model_file =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metal-conv.onnx");
        let mut model = tract_onnx::onnx()
            .model_for_path(&model_file)?
            .with_input_fact(0, f32::fact([1, 3, 2, 2]).into())?
            .into_typed()?;
        tract_metal::MetalTransform::default().transform(&mut model)?;
        if !model
            .nodes()
            .iter()
            .any(TypedNode::op_is::<tract_metal::ops::conv::MetalConv>)
        {
            return Err(std::io::Error::other(
                "fixture convolution was not translated to Tract's stable MetalConv operation",
            )
            .into());
        }
        Ok(())
    }
}
