use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gst::{glib, prelude::*, subclass::prelude::*};
use gst_base::subclass::prelude::*;
use gst_video::prelude::*;

use crate::engine::Engine;
use gst_inference_common::model_info::ModelInfo;
use gst_inference_common::preprocess::{PixelFormat, preprocess};
use gst_inference_common::tensor;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "tractinference",
        gst::DebugColorFlags::empty(),
        Some("Tract inference element"),
    )
});

#[derive(Default)]
struct Settings {
    model_file: Option<PathBuf>,
    model_info_file: Option<PathBuf>,
}

struct State {
    engine: Box<dyn Engine>,
    info: ModelInfo,
    video_info: Option<gst_video::VideoInfo>,
}

#[derive(Default)]
pub struct TractInference {
    state: Mutex<Option<State>>,
    settings: Mutex<Settings>,
}

#[glib::object_subclass]
impl ObjectSubclass for TractInference {
    const NAME: &'static str = "GstSmithTractInference";
    type Type = super::TractInference;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for TractInference {
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
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let Ok(mut settings) = self.settings.lock() else {
            return;
        };
        let path: Option<String> = match value.get() {
            Ok(path) => path,
            Err(_) => return,
        };
        match pspec.name() {
            "model-file" => settings.model_file = path.map(PathBuf::from),
            "model-info-file" => settings.model_info_file = path.map(PathBuf::from),
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
            _ => None::<String>.to_value(),
        }
    }
}

impl GstObjectImpl for TractInference {}

impl ElementImpl for TractInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Tract ONNX Inference",
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

impl BaseTransformImpl for TractInference {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        #[cfg(not(feature = "tract"))]
        return Err(gst::error_msg!(
            gst::LibraryError::Settings,
            ["tract backend is disabled at compile time"]
        ));
        #[cfg(feature = "tract")]
        {
            let settings = self.settings.lock().map_err(|_error| {
                gst::error_msg!(
                    gst::LibraryError::Settings,
                    ["inference settings lock is poisoned"]
                )
            })?;
            let model_file = settings.model_file.clone().ok_or_else(|| {
                gst::error_msg!(
                    gst::LibraryError::Settings,
                    ["model-file must be set before starting tractinference"]
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
            let engine: Box<dyn Engine> = Box::new(
                crate::engine::tract::TractEngine::load(&model_file, &info).map_err(|error| {
                    gst::error_msg!(
                        gst::LibraryError::Settings,
                        [
                            "failed to initialize Tract model {}: {error}",
                            model_file.display()
                        ]
                    )
                })?,
            );
            drop(settings);
            let mut state = self.state.lock().map_err(|_error| {
                gst::error_msg!(
                    gst::LibraryError::Settings,
                    ["inference state lock is poisoned"]
                )
            })?;
            *state = Some(State {
                engine,
                info,
                video_info: None,
            });
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
        {
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
        }
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
                ["Tract inference failed: {error}"]
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

#[cfg(test)]
mod tests {
    use std::fs;

    use gst::glib;
    use gst::glib::subclass::prelude::ObjectSubclassIsExt;
    use gst_base::subclass::prelude::BaseTransformImpl;

    fn started_element()
    -> Result<(super::super::TractInference, tempfile::TempDir), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let model = directory.path().join("identity.onnx");
        fs::write(
            &model,
            include_bytes!("../../../inference-common/tests/fixtures/identity.onnx"),
        )?;
        fs::write(
            directory.path().join("identity.onnx.modelinfo"),
            include_str!("../../../inference-common/tests/fixtures/identity.onnx.modelinfo"),
        )?;
        let element: super::super::TractInference = glib::Object::builder()
            .property("model-file", model.to_string_lossy().as_ref())
            .build();
        BaseTransformImpl::start(element.imp()).map_err(std::io::Error::other)?;
        Ok((element, directory))
    }

    #[test]
    #[expect(
        clippy::panic_in_result_fn,
        reason = "GStreamer caps assertions make an interoperability regression immediately visible"
    )]
    fn transforms_caps_without_synthesizing_reverse_tensor_groups()
    -> Result<(), Box<dyn std::error::Error>> {
        gst::init()?;
        let (element, _directory) = started_element()?;
        let imp = element.imp();
        let base = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build();
        let forward = BaseTransformImpl::transform_caps(imp, gst::PadDirection::Sink, &base, None)
            .ok_or_else(|| std::io::Error::other("forward caps transform failed"))?;
        let forward_structure = forward
            .structure(0)
            .ok_or_else(|| std::io::Error::other("missing forward caps structure"))?;
        assert_eq!(forward_structure.get::<i32>("width")?, 2);
        assert_eq!(forward_structure.get::<i32>("height")?, 1);
        let forward_groups = forward_structure.get::<gst::Structure>("tensors")?;
        let fixture_group = forward_groups.get::<gst::UniqueList>("gstsmith-identity-fixture")?;
        assert_eq!(fixture_group.as_slice().len(), 2);

        let reverse_without_groups =
            BaseTransformImpl::transform_caps(imp, gst::PadDirection::Src, &base, None)
                .ok_or_else(|| std::io::Error::other("reverse caps transform failed"))?;
        assert!(
            !reverse_without_groups
                .structure(0)
                .ok_or_else(|| std::io::Error::other("missing reverse caps structure"))?
                .has_field("tensors")
        );

        let mut groups = gst::Structure::new_empty("tensorgroups");
        groups.set("unrelated", gst::List::new([gst::Caps::new_any()]));
        groups.set(
            "gstsmith-identity-fixture",
            gst::List::new([gst::Caps::new_any()]),
        );
        let with_groups = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("tensors", groups)
            .build();
        let reverse_with_groups =
            BaseTransformImpl::transform_caps(imp, gst::PadDirection::Src, &with_groups, None)
                .ok_or_else(|| std::io::Error::other("reverse grouped caps transform failed"))?;
        let groups = reverse_with_groups
            .structure(0)
            .ok_or_else(|| std::io::Error::other("missing grouped reverse caps structure"))?
            .get::<gst::Structure>("tensors")?;
        assert!(groups.has_field("unrelated"));
        assert!(!groups.has_field("gstsmith-identity-fixture"));

        BaseTransformImpl::stop(imp).map_err(std::io::Error::other)?;
        Ok(())
    }
}
