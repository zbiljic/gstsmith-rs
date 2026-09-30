use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gst::{glib, prelude::*, subclass::prelude::*};
use gst_base::subclass::prelude::*;

use crate::engine::TensorEngine;
use crate::tractinference::imp::ExecutionProvider;
use gst_inference_common::model_info::TensorModelInfo;
use gst_inference_common::tensor;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "tracttensorinference",
        gst::DebugColorFlags::empty(),
        Some("Tract tensor-input inference element"),
    )
});

#[derive(Default)]
struct Settings {
    model_file: Option<PathBuf>,
    model_info_file: Option<PathBuf>,
    execution_provider: ExecutionProvider,
}

struct State {
    engine: Box<dyn TensorEngine>,
    info: TensorModelInfo,
}

#[derive(Default)]
pub struct TractTensorInference {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
}

fn settings_error() -> gst::ErrorMessage {
    gst::error_msg!(
        gst::LibraryError::Settings,
        ["inference settings lock is poisoned"]
    )
}

#[glib::object_subclass]
impl ObjectSubclass for TractTensorInference {
    const NAME: &'static str = "GstSmithTractTensorInference";
    type Type = super::TractTensorInference;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for TractTensorInference {
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
                    .blurb("Tract execution provider")
                    .default_value(ExecutionProvider::Cpu)
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
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for TractTensorInference {}

impl ElementImpl for TractTensorInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Tract ONNX Tensor Inference",
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

impl BaseTransformImpl for TractTensorInference {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().map_err(|_error| settings_error())?;
        let execution_provider = settings.execution_provider;
        if execution_provider == ExecutionProvider::Metal {
            #[cfg(not(target_os = "macos"))]
            return Err(gst::error_msg!(
                gst::LibraryError::Settings,
                ["Metal execution is only supported on macOS"]
            ));
            #[cfg(all(target_os = "macos", not(feature = "metal")))]
            return Err(gst::error_msg!(
                gst::LibraryError::Settings,
                ["Metal support was not compiled; rebuild with the `metal` feature"]
            ));
        }
        #[cfg(not(feature = "tract"))]
        return Err(gst::error_msg!(
            gst::LibraryError::Settings,
            ["tract backend is disabled at compile time"]
        ));
        #[cfg(feature = "tract")]
        {
            let model_file = settings.model_file.clone().ok_or_else(|| {
                gst::error_msg!(
                    gst::LibraryError::Settings,
                    ["model-file must be set before starting tracttensorinference"]
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
            let engine: Box<dyn TensorEngine> = Box::new(
                crate::engine::tract::TractTensorEngine::load(
                    &model_file,
                    &info,
                    execution_provider,
                )
                .map_err(|error| {
                    gst::error_msg!(
                        gst::LibraryError::Settings,
                        [
                            "failed to initialize Tract model {}: {error}",
                            model_file.display()
                        ]
                    )
                })?,
            );
            let mut state = self.state.lock().map_err(|_error| {
                gst::error_msg!(
                    gst::LibraryError::Settings,
                    ["inference state lock is poisoned"]
                )
            })?;
            *state = Some(State { engine, info });
            Ok(())
        }
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
                ["Tract inference failed: {error}"]
            );
            gst::FlowError::Error
        })?;
        tensor::attach_tensors(buffer, outputs);
        Ok(gst::FlowSuccess::Ok)
    }
}
