use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gst::{glib, prelude::*, subclass::prelude::*};
use gst_base::subclass::prelude::*;
use gst_video::prelude::*;

use crate::engine::{EngineOptions, OrtEngine, Provider};
use gst_inference_common::model_info::ModelInfo;
use gst_inference_common::preprocess::{ChannelOrder, PixelFormat, preprocess};
use gst_inference_common::tensor;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "ortinference",
        gst::DebugColorFlags::empty(),
        Some("ONNX Runtime inference element"),
    )
});

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtExecutionProvider")]
pub enum ExecutionProvider {
    #[default]
    #[enum_value(name = "CPU", nick = "cpu")]
    Cpu = 0,
    #[cfg(feature = "coreml")]
    #[enum_value(name = "CoreML", nick = "coreml")]
    Coreml = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtModelChannelOrder")]
pub enum ModelChannelOrder {
    #[default]
    #[enum_value(name = "RGB", nick = "rgb")]
    Rgb = 0,
    #[enum_value(name = "BGR", nick = "bgr")]
    Bgr = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithOrtGraphOptimization")]
pub enum GraphOptimization {
    Disable = 0,
    Level1 = 1,
    Level2 = 2,
    #[default]
    Level3 = 3,
    All = 4,
}

#[derive(Default)]
struct Settings {
    model_file: Option<PathBuf>,
    model_info_file: Option<PathBuf>,
    execution_provider: ExecutionProvider,
    intra_threads: Option<u32>,
    optimization: GraphOptimization,
    strict_execution_provider: bool,
    model_channel_order: ModelChannelOrder,
}

struct State {
    engine: Box<dyn gst_inference_common::engine::Engine>,
    info: ModelInfo,
    video_info: Option<gst_video::VideoInfo>,
    channel_order: ChannelOrder,
}

#[derive(Default)]
pub struct OrtInference {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
}

#[glib::object_subclass]
impl ObjectSubclass for OrtInference {
    const NAME: &'static str = "GstSmithOrtInference";
    type Type = super::OrtInference;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for OrtInference {
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
                glib::ParamSpecEnum::builder::<ModelChannelOrder>("model-channel-order")
                    .nick("Model Channel Order")
                    .blurb("Channel order expected by the model input tensor")
                    .default_value(ModelChannelOrder::Rgb)
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
            "model-channel-order" => {
                if let Ok(order) = value.get::<ModelChannelOrder>() {
                    settings.model_channel_order = order;
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
            "model-channel-order" => settings.model_channel_order.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for OrtInference {}

impl ElementImpl for OrtInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "ONNX Runtime Inference",
                "Filter/Analysis/Video",
                "Runs a model-agnostic ONNX image model and attaches output tensors",
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
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", gst::List::new(["RGB", "BGR", "RGBA", "BGRA"]))
                .build();
            let sink = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("construct inference sink template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("construct inference src template");
            vec![sink, src]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for OrtInference {
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
        let provider = match settings.execution_provider {
            ExecutionProvider::Cpu => Provider::Cpu,
            #[cfg(feature = "coreml")]
            ExecutionProvider::Coreml => Provider::Coreml,
        };
        let threads = settings
            .intra_threads
            .map(|value| {
                usize::try_from(value).map_err(|_error| {
                    gst::error_msg!(
                        gst::LibraryError::Settings,
                        ["intra-op-threads does not fit the platform usize"]
                    )
                })
            })
            .transpose()?;
        let optimization = match settings.optimization {
            GraphOptimization::Disable => ort::session::builder::GraphOptimizationLevel::Disable,
            GraphOptimization::Level1 => ort::session::builder::GraphOptimizationLevel::Level1,
            GraphOptimization::Level2 => ort::session::builder::GraphOptimizationLevel::Level2,
            GraphOptimization::Level3 => ort::session::builder::GraphOptimizationLevel::Level3,
            GraphOptimization::All => ort::session::builder::GraphOptimizationLevel::All,
        };
        let options = EngineOptions {
            provider,
            intra_threads: threads,
            optimization,
            strict_execution_provider: settings.strict_execution_provider,
        }
        .validate()
        .map_err(|error| gst::error_msg!(gst::LibraryError::Settings, ["{error}"]))?;
        let model_file = settings.model_file.clone().ok_or_else(|| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["model-file must be set before starting ortinference"]
            )
        })?;
        let info_file = settings
            .model_info_file
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{}.modelinfo", model_file.display())));
        let contents = std::fs::read_to_string(&info_file).map_err(|error| {
            gst::error_msg!(
                gst::ResourceError::OpenRead,
                [
                    "failed to read model-info file {}: {error}",
                    info_file.display()
                ]
            )
        })?;
        let info = ModelInfo::parse(&contents).map_err(|error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["invalid model-info file {}: {error}", info_file.display()]
            )
        })?;
        let channel_order = match settings.model_channel_order {
            ModelChannelOrder::Rgb => ChannelOrder::Rgb,
            ModelChannelOrder::Bgr => ChannelOrder::Bgr,
        };
        let engine = OrtEngine::load(&model_file, &info, options).map_err(|error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                [
                    "failed to initialize ONNX Runtime model {}: {error}",
                    model_file.display()
                ]
            )
        })?;
        drop(settings);
        let mut state = self.state.lock().map_err(|_error| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["inference state lock is poisoned"]
            )
        })?;
        *state = Some(State {
            engine: Box::new(engine),
            info,
            video_info: None,
            channel_order,
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

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let video_info = gst_video::VideoInfo::from_caps(incaps)
            .map_err(|_error| gst::loggable_error!(CAT, "invalid video caps {incaps:?}"))?;
        if pixel_format(video_info.format()).is_none() {
            return Err(gst::loggable_error!(
                CAT,
                "unsupported negotiated video format {:?}",
                video_info.format()
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_error| gst::loggable_error!(CAT, "inference state lock is poisoned"))?;
        let Some(state) = state.as_mut() else {
            return Err(gst::loggable_error!(CAT, "inference engine is not started"));
        };
        let expected = state.info.image_dimensions().map_err(|error| {
            gst::loggable_error!(CAT, "invalid model input dimensions: {error}")
        })?;
        let actual = (video_info.width() as usize, video_info.height() as usize);
        if actual != expected {
            return Err(gst::loggable_error!(
                CAT,
                "negotiated video size {}x{} does not match model input {}x{}",
                actual.0,
                actual.1,
                expected.0,
                expected.1
            ));
        }
        state.video_info = Some(video_info);
        self.parent_set_caps(incaps, outcaps)
    }

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let state = self.state.lock().ok();
        let info = state
            .as_deref()
            .and_then(Option::as_ref)
            .map(|state| &state.info);
        Some(tensor::transform_caps(info, direction, caps, filter))
    }

    fn fixate_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        othercaps: gst::Caps,
    ) -> gst::Caps {
        let state = self.state.lock().ok();
        let info = state
            .as_deref()
            .and_then(Option::as_ref)
            .map(|state| &state.info);
        tensor::fixate_caps(info, direction, caps, othercaps)
    }

    fn transform_ip(
        &self,
        buffer: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let state = self.state.lock().map_err(|_error| gst::FlowError::Error)?;
        let Some(state) = state.as_ref() else {
            return Err(gst::FlowError::Flushing);
        };
        let info = state
            .video_info
            .as_ref()
            .ok_or(gst::FlowError::NotNegotiated)?;
        let frame =
            gst_video::VideoFrameRef::from_buffer_ref_writable(buffer, info).map_err(|error| {
                gst::element_imp_error!(
                    self,
                    gst::ResourceError::Read,
                    ["failed to map video frame: {error}"]
                );
                gst::FlowError::Error
            })?;
        let format = pixel_format(frame.format()).ok_or_else(|| {
            gst::element_imp_error!(
                self,
                gst::StreamError::Format,
                ["unsupported negotiated video format {:?}", frame.format()]
            );
            gst::FlowError::NotNegotiated
        })?;
        let stride = frame
            .plane_stride()
            .first()
            .copied()
            .ok_or(gst::FlowError::Error)
            .and_then(|stride| usize::try_from(stride).map_err(|_error| gst::FlowError::Error))?;
        let data = frame.plane_data(0).map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Read,
                ["failed to access video plane: {error}"]
            );
            gst::FlowError::Error
        })?;
        let input = preprocess(
            data,
            stride,
            frame.width() as usize,
            frame.height() as usize,
            format,
            state.channel_order,
            state.info.input(),
        )
        .map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::StreamError::Format,
                ["failed to preprocess frame: {error}"]
            );
            gst::FlowError::Error
        })?;
        drop(frame);
        let outputs = state.engine.run(input).map_err(|error| {
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

fn pixel_format(format: gst_video::VideoFormat) -> Option<PixelFormat> {
    match format {
        gst_video::VideoFormat::Rgb => Some(PixelFormat::Rgb),
        gst_video::VideoFormat::Bgr => Some(PixelFormat::Bgr),
        gst_video::VideoFormat::Rgba => Some(PixelFormat::Rgba),
        gst_video::VideoFormat::Bgra => Some(PixelFormat::Bgra),
        _ => None,
    }
}
