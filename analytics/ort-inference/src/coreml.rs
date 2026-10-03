use gst::{glib, prelude::*};

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtCoreMlModelFormat")]
pub enum ModelFormat {
    #[default]
    #[enum_value(name = "NeuralNetwork", nick = "neural-network")]
    NeuralNetwork = 0,
    #[enum_value(name = "MLProgram", nick = "mlprogram")]
    MlProgram = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtCoreMlComputeUnits")]
pub enum ComputeUnits {
    #[default]
    #[enum_value(name = "All", nick = "all")]
    All = 0,
    #[enum_value(name = "CPU only", nick = "cpu-only")]
    CpuOnly = 1,
    #[enum_value(name = "CPU and GPU", nick = "cpu-and-gpu")]
    CpuAndGpu = 2,
    #[enum_value(name = "CPU and Neural Engine", nick = "cpu-and-neural-engine")]
    CpuAndNeuralEngine = 3,
}

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtCoreMlSpecializationStrategy")]
pub enum SpecializationStrategy {
    #[default]
    Default = 0,
    #[enum_value(name = "Fast prediction", nick = "fast-prediction")]
    FastPrediction = 1,
}

/// Independent `CoreML` provider options, with ONNX Runtime's defaults.
#[derive(Clone, Debug, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "these are independent ONNX Runtime provider switches, not mutually exclusive states"
)]
pub struct CoreMlOptions {
    model_format: ModelFormat,
    compute_units: ComputeUnits,
    require_static_input_shapes: bool,
    model_cache_directory: Option<String>,
    specialization_strategy: SpecializationStrategy,
    profile_compute_plan: bool,
    enable_on_subgraphs: bool,
    allow_low_precision_accumulation_on_gpu: bool,
}

impl CoreMlOptions {
    pub fn properties() -> [glib::ParamSpec; 8] {
        [
            glib::ParamSpecEnum::builder::<ModelFormat>("coreml-model-format")
                .nick("CoreML Model Format")
                .blurb("CoreML model format; MLProgram requires macOS 12+ or iOS 15+")
                .mutable_ready()
                .build(),
            glib::ParamSpecEnum::builder::<ComputeUnits>("coreml-compute-units")
                .nick("CoreML Compute Units")
                .blurb("Devices CoreML may use; does not guarantee Neural Engine execution")
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("coreml-require-static-input-shapes")
                .nick("CoreML Require Static Input Shapes")
                .blurb("Only assign nodes with static input shapes to CoreML; does not specialize the model")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("coreml-model-cache-directory")
                .nick("CoreML Model Cache Directory")
                .blurb("Compiled-model cache directory; unset or empty disables caching; caller must invalidate stale entries")
                .mutable_ready()
                .build(),
            glib::ParamSpecEnum::builder::<SpecializationStrategy>("coreml-specialization-strategy")
                .nick("CoreML Specialization Strategy")
                .blurb("Fast prediction may increase model loading time and resource usage")
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("coreml-profile-compute-plan")
                .nick("CoreML Profile Compute Plan")
                .blurb("Log CoreML operator hardware placement and estimated execution time")
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("coreml-enable-on-subgraphs")
                .nick("CoreML Enable on Subgraphs")
                .blurb("Allow CoreML in Loop, Scan, and If subgraphs")
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("coreml-allow-low-precision-accumulation-on-gpu")
                .nick("CoreML Allow Low Precision Accumulation on GPU")
                .blurb("Allow FP16 accumulation on GPU, potentially reducing numerical accuracy")
                .mutable_ready()
                .build(),
        ]
    }

    pub fn set_property(&mut self, value: &glib::Value, pspec: &glib::ParamSpec) -> bool {
        match pspec.name() {
            "coreml-model-format" => {
                if let Ok(format) = value.get() {
                    self.model_format = format;
                }
            }
            "coreml-compute-units" => {
                if let Ok(units) = value.get() {
                    self.compute_units = units;
                }
            }
            "coreml-require-static-input-shapes" => {
                if let Ok(enabled) = value.get() {
                    self.require_static_input_shapes = enabled;
                }
            }
            "coreml-model-cache-directory" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    self.model_cache_directory = path.filter(|path| !path.is_empty());
                }
            }
            "coreml-specialization-strategy" => {
                if let Ok(strategy) = value.get() {
                    self.specialization_strategy = strategy;
                }
            }
            "coreml-profile-compute-plan" => {
                if let Ok(enabled) = value.get() {
                    self.profile_compute_plan = enabled;
                }
            }
            "coreml-enable-on-subgraphs" => {
                if let Ok(enabled) = value.get() {
                    self.enable_on_subgraphs = enabled;
                }
            }
            "coreml-allow-low-precision-accumulation-on-gpu" => {
                if let Ok(enabled) = value.get() {
                    self.allow_low_precision_accumulation_on_gpu = enabled;
                }
            }
            _ => return false,
        }
        true
    }

    pub fn property(&self, name: &str) -> Option<glib::Value> {
        Some(match name {
            "coreml-model-format" => self.model_format.to_value(),
            "coreml-compute-units" => self.compute_units.to_value(),
            "coreml-require-static-input-shapes" => self.require_static_input_shapes.to_value(),
            "coreml-model-cache-directory" => self.model_cache_directory.to_value(),
            "coreml-specialization-strategy" => self.specialization_strategy.to_value(),
            "coreml-profile-compute-plan" => self.profile_compute_plan.to_value(),
            "coreml-enable-on-subgraphs" => self.enable_on_subgraphs.to_value(),
            "coreml-allow-low-precision-accumulation-on-gpu" => {
                self.allow_low_precision_accumulation_on_gpu.to_value()
            }
            _ => return None,
        })
    }

    pub fn execution_provider(&self) -> ort::ep::CoreML {
        let format = match self.model_format {
            ModelFormat::NeuralNetwork => ort::ep::coreml::ModelFormat::NeuralNetwork,
            ModelFormat::MlProgram => ort::ep::coreml::ModelFormat::MLProgram,
        };
        let units = match self.compute_units {
            ComputeUnits::All => ort::ep::coreml::ComputeUnits::All,
            ComputeUnits::CpuOnly => ort::ep::coreml::ComputeUnits::CPUOnly,
            ComputeUnits::CpuAndGpu => ort::ep::coreml::ComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine,
        };
        let strategy = match self.specialization_strategy {
            SpecializationStrategy::Default => ort::ep::coreml::SpecializationStrategy::Default,
            SpecializationStrategy::FastPrediction => {
                ort::ep::coreml::SpecializationStrategy::FastPrediction
            }
        };
        let provider = ort::ep::CoreML::default()
            .with_model_format(format)
            .with_compute_units(units)
            .with_static_input_shapes(self.require_static_input_shapes)
            .with_specialization_strategy(strategy)
            .with_profile_compute_plan(self.profile_compute_plan)
            .with_subgraphs(self.enable_on_subgraphs)
            .with_low_precision_accumulation_on_gpu(self.allow_low_precision_accumulation_on_gpu);
        match &self.model_cache_directory {
            Some(path) => provider.with_model_cache_dir(path),
            None => provider,
        }
    }
}
