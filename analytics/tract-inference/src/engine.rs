pub use gst_inference_common::engine::Engine;
#[cfg(feature = "tract")]
pub use gst_inference_common::engine::{InputTensor, OwnedTensor};

#[cfg(feature = "tract")]
pub mod tract {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::infer::Factoid;

    use super::{Engine, InputTensor, OwnedTensor};
    use crate::tractinference::imp::ExecutionProvider;
    use gst_inference_common::model_info::{ModelInfo, ScalarType, TensorDescription};

    pub struct TractEngine {
        plan: std::sync::Arc<TypedRunnableModel>,
        input: TensorDescription,
        outputs: Vec<TensorDescription>,
    }

    impl TractEngine {
        pub fn load(
            model_file: &std::path::Path,
            info: &ModelInfo,
            execution_provider: ExecutionProvider,
        ) -> Result<Self, String> {
            let model = tract_onnx::onnx()
                .model_for_path(model_file)
                .map_err(|error| format!("failed to load ONNX model: {error}"))?;
            let input = info.input().clone();
            let runtime_inputs = model
                .input_outlets()
                .map_err(|error| format!("failed to inspect model inputs: {error}"))?;
            if runtime_inputs.len() != 1 {
                return Err(format!(
                    "model has {} inputs; exactly one is supported",
                    runtime_inputs.len()
                ));
            }
            let runtime_input = runtime_inputs
                .first()
                .copied()
                .ok_or_else(|| "model did not expose its input".to_owned())?;
            validate_name(outlet_name(&model, runtime_input), &input.name, "input", 0)?;
            let runtime_input_fact = model
                .input_fact(0)
                .map_err(|error| format!("failed to inspect model input: {error}"))?;
            validate_fact(runtime_input_fact, &input, "input", 0)?;
            let declared_outputs = model
                .output_outlets()
                .map_err(|error| format!("failed to inspect model outputs: {error}"))?;
            if declared_outputs.len() != info.outputs().len() {
                return Err(format!(
                    "model has {} outputs but model-info declares {}",
                    declared_outputs.len(),
                    info.outputs().len()
                ));
            }
            for (index, (outlet, descriptor)) in
                declared_outputs.iter().zip(info.outputs()).enumerate()
            {
                validate_name(
                    outlet_name(&model, *outlet),
                    &descriptor.name,
                    "output",
                    index,
                )?;
            }
            let fact: InferenceFact = match input.data_type {
                ScalarType::Float32 => f32::fact(input.dims.clone()).into(),
                ScalarType::Uint8 => u8::fact(input.dims.clone()).into(),
                _ => return Err("model-info permits only float32 or uint8 inputs".to_owned()),
            };
            let mut model = model
                .with_input_fact(0, fact)
                .map_err(|error| format!("failed to specialize model input: {error}"))?
                .into_typed()
                .map_err(|error| format!("failed to convert model to a typed graph: {error}"))?;
            apply_execution_provider(&mut model, execution_provider)?;
            let model = model
                .into_optimized()
                .map_err(|error| format!("failed to optimize model: {error}"))?;

            let runtime_outputs = model
                .output_outlets()
                .map_err(|error| format!("failed to inspect model outputs: {error}"))?;
            if runtime_outputs.len() != info.outputs().len() {
                return Err(format!(
                    "model has {} outputs but model-info declares {}",
                    runtime_outputs.len(),
                    info.outputs().len()
                ));
            }
            for (index, descriptor) in info.outputs().iter().enumerate() {
                let fact = model
                    .output_fact(index)
                    .map_err(|error| format!("failed to inspect output {index}: {error}"))?;
                validate_typed_fact(fact, descriptor, "output", index)?;
            }
            let plan = model
                .into_runnable()
                .map_err(|error| format!("failed to create runnable model: {error}"))?;
            Ok(Self {
                plan,
                input,
                outputs: info.outputs().to_vec(),
            })
        }
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
            let runtime_outputs = self
                .plan
                .run(tvec![tensor.into()])
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

    fn validate_name(
        discovered: Option<&str>,
        expected: &str,
        direction: &str,
        index: usize,
    ) -> Result<(), String> {
        if discovered == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "{direction} {index} name mismatch: model {discovered:?}, model-info {expected:?}"
            ))
        }
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
        let shape = fact
            .shape
            .as_concrete_finite()
            .map_err(|error| format!("failed to inspect {direction} {index} dimensions: {error}"))?
            .ok_or_else(|| format!("{direction} {index} has dynamic dimensions"))?;
        if shape.as_slice() != descriptor.dims.as_slice() {
            return Err(format!(
                "{direction} {index} dimensions mismatch: model {shape:?}, model-info {:?}",
                descriptor.dims
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
        let values: Result<Vec<_>, _> = first
            .bytes
            .chunks_exact(4)
            .map(|chunk| <[u8; 4]>::try_from(chunk).map(f32::from_le_bytes))
            .collect();
        if values? != [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
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
