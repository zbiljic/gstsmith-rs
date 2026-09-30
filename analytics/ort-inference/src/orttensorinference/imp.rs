use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gst::{glib, prelude::*, subclass::prelude::*};
use gst_base::subclass::prelude::*;

use crate::engine::OrtTensorEngine;
use crate::ortinference::imp::{ExecutionProvider, GraphOptimization, engine_options};
use gst_inference_common::engine::TensorEngine;
use gst_inference_common::model_info::TensorModelInfo;
use gst_inference_common::tensor;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "orttensorinference",
        gst::DebugColorFlags::empty(),
        Some("ONNX Runtime tensor-input inference element"),
    )
});

#[derive(Default)]
struct Settings {
    model_file: Option<PathBuf>,
    model_info_file: Option<PathBuf>,
    execution_provider: ExecutionProvider,
    intra_threads: Option<u32>,
    optimization: GraphOptimization,
    strict_execution_provider: bool,
}

struct State {
    engine: Box<dyn TensorEngine>,
    info: TensorModelInfo,
}

#[derive(Default)]
pub struct OrtTensorInference {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
}

#[glib::object_subclass]
impl ObjectSubclass for OrtTensorInference {
    const NAME: &'static str = "GstSmithOrtTensorInference";
    type Type = super::OrtTensorInference;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for OrtTensorInference {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("model-file")
                    .nick("Model File")
                    .blurb("ONNX model file")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("model-info-file")
                    .nick("Model Info File")
                    .blurb("Optional model-info file override")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecEnum::builder::<ExecutionProvider>("execution-provider")
                    .nick("Execution Provider")
                    .blurb("ONNX Runtime execution provider")
                    .default_value(ExecutionProvider::Cpu)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("intra-op-threads")
                    .nick("Intra-op Threads")
                    .blurb("Positive ONNX Runtime intra-op thread count; zero uses ORT policy")
                    .default_value(0)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecEnum::builder::<GraphOptimization>("graph-optimization")
                    .nick("Graph Optimization")
                    .blurb("ONNX Runtime graph optimization level")
                    .default_value(GraphOptimization::Level3)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("strict-execution-provider")
                    .nick("Strict Execution Provider")
                    .blurb("Disable ONNX Runtime CPU fallback for a non-CPU provider")
                    .default_value(false)
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let Ok(mut settings) = self.settings.lock() else {
            return;
        };
        match pspec.name() {
            "model-file" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    settings.model_file = path.map(PathBuf::from);
                }
            }
            "model-info-file" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    settings.model_info_file = path.map(PathBuf::from);
                }
            }
            "execution-provider" => {
                if let Ok(provider) = value.get::<ExecutionProvider>() {
                    settings.execution_provider = provider;
                }
            }
            "intra-op-threads" => {
                if let Ok(threads) = value.get::<u32>() {
                    settings.intra_threads = (threads != 0).then_some(threads);
                }
            }
            "graph-optimization" => {
                if let Ok(level) = value.get::<GraphOptimization>() {
                    settings.optimization = level;
                }
            }
            "strict-execution-provider" => {
                if let Ok(enabled) = value.get::<bool>() {
                    settings.strict_execution_provider = enabled;
                }
            }
            _ => gst::warning!(CAT, imp = self, "unexpected property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let Ok(settings) = self.settings.lock() else {
            return None::<String>.to_value();
        };
        match pspec.name() {
            "model-file" => settings
                .model_file
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .to_value(),
            "model-info-file" => settings
                .model_info_file
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .to_value(),
            "execution-provider" => settings.execution_provider.to_value(),
            "intra-op-threads" => settings.intra_threads.unwrap_or(0).to_value(),
            "graph-optimization" => settings.optimization.to_value(),
            "strict-execution-provider" => settings.strict_execution_provider.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for OrtTensorInference {}

impl ElementImpl for OrtTensorInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "ONNX Runtime Tensor Inference",
                "Filter/Analysis",
                "Runs a model-agnostic ONNX model on upstream tensors and attaches output tensors",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });
        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "static pad-template construction has fixed valid names and caps"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = gst::Caps::new_any();
            let sink = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("construct tensor inference sink template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("construct tensor inference src template");
            vec![sink, src]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for OrtTensorInference {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().map_err(|_error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["inference settings lock is poisoned"]
            )
        })?;
        let options = engine_options(
            settings.execution_provider,
            settings.intra_threads,
            settings.optimization,
            settings.strict_execution_provider,
        )?;
        let model_file = settings.model_file.clone().ok_or_else(|| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["model-file must be set before starting orttensorinference"]
            )
        })?;
        let info_file = settings
            .model_info_file
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{}.modelinfo", model_file.display())));
        drop(settings);
        let contents = std::fs::read_to_string(&info_file).map_err(|error| {
            gst::error_msg!(
                gst::ResourceError::OpenRead,
                [
                    "failed to read model-info file {}: {error}",
                    info_file.display()
                ]
            )
        })?;
        let info = TensorModelInfo::parse(&contents).map_err(|error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["invalid model-info file {}: {error}", info_file.display()]
            )
        })?;
        let engine = OrtTensorEngine::load(&model_file, &info, options).map_err(|error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                [
                    "failed to initialize ONNX Runtime model {}: {error}",
                    model_file.display()
                ]
            )
        })?;
        let mut state = self.state.lock().map_err(|_error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["inference state lock is poisoned"]
            )
        })?;
        *state = Some(State {
            engine: Box::new(engine),
            info,
        });
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().map_err(|_error| {
            gst::error_msg!(
                gst::LibraryError::Failed,
                ["inference state lock is poisoned"]
            )
        })?;
        *state = None;
        Ok(())
    }

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let state = self.state.lock().ok();
        let contract = state
            .as_deref()
            .and_then(Option::as_ref)
            .map(|state| state.info.caps_contract());
        Some(tensor::transform_caps(contract, direction, caps, filter))
    }

    fn fixate_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        othercaps: gst::Caps,
    ) -> gst::Caps {
        let state = self.state.lock().ok();
        let contract = state
            .as_deref()
            .and_then(Option::as_ref)
            .map(|state| state.info.caps_contract());
        tensor::fixate_caps(contract, direction, caps, othercaps)
    }

    fn transform_ip(
        &self,
        buffer: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let state = self.state.lock().map_err(|_error| gst::FlowError::Error)?;
        let Some(state) = state.as_ref() else {
            return Err(gst::FlowError::Flushing);
        };
        let inputs = match tensor::collect_inputs(buffer, &state.info) {
            Ok(Some(inputs)) => inputs,
            Ok(None) => return Ok(gst::FlowSuccess::Ok),
            Err(error) => {
                gst::element_imp_error!(
                    self,
                    gst::StreamError::Format,
                    ["invalid model inputs: {error}"]
                );
                return Err(gst::FlowError::Error);
            }
        };
        let outputs = state.engine.run(&inputs).map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::StreamError::Failed,
                ["ONNX Runtime inference failed: {error}"]
            );
            gst::FlowError::Error
        })?;
        tensor::attach_tensors(buffer, outputs);
        Ok(gst::FlowSuccess::Ok)
    }
}
