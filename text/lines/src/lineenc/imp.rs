use std::sync::{LazyLock, Mutex, MutexGuard};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;

const DEFAULT_DELIMITER: &str = "\n";

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "lineenc",
        gst::DebugColorFlags::empty(),
        Some("Delimiter encoder"),
    )
});

struct Settings {
    delimiter: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delimiter: DEFAULT_DELIMITER.to_owned(),
        }
    }
}

#[derive(Default)]
pub struct LineEnc {
    settings: Mutex<Settings>,
}

impl LineEnc {
    fn settings(&self) -> MutexGuard<'_, Settings> {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[glib::object_subclass]
impl ObjectSubclass for LineEnc {
    const NAME: &'static str = "GstSmithLineEnc";
    type Type = super::LineEnc;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for LineEnc {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("delimiter")
                    .nick("Delimiter")
                    .blurb(
                        "Non-empty delimiter appended to records that do not already end with it",
                    )
                    .default_value(Some(DEFAULT_DELIMITER))
                    .mutable_ready()
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        if pspec.name() == "delimiter" {
            if let Ok(delimiter) = value.get::<String>() {
                self.settings().delimiter = delimiter;
            }
        } else {
            gst::warning!(CAT, imp = self, "Unknown property {}", pspec.name());
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        if pspec.name() == "delimiter" {
            self.settings().delimiter.to_value()
        } else {
            pspec.default_value().clone()
        }
    }
}

impl GstObjectImpl for LineEnc {}

impl ElementImpl for LineEnc {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Delimiter Encoder",
                "Codec/Encoder/Text",
                "Ensures every input record ends with a configured delimiter",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });

        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "constant pad names and ANY caps make template construction infallible"
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
            .expect("constructing the lineenc sink pad template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("constructing the lineenc source pad template");

            vec![sink, src]
        });

        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for LineEnc {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        if self.settings().delimiter.is_empty() {
            return Err(gst::error_msg!(
                gst::CoreError::StateChange,
                ["delimiter must not be empty"]
            ));
        }
        Ok(())
    }

    fn transform_caps(
        &self,
        _direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let caps = caps.clone();
        Some(if let Some(filter) = filter {
            filter.intersect_with_mode(&caps, gst::CapsIntersectMode::First)
        } else {
            caps
        })
    }

    fn transform_size(
        &self,
        direction: gst::PadDirection,
        _caps: &gst::Caps,
        size: usize,
        _othercaps: &gst::Caps,
    ) -> Option<usize> {
        match direction {
            gst::PadDirection::Sink => size.checked_add(self.settings().delimiter.len()),
            gst::PadDirection::Src => Some(size),
            gst::PadDirection::Unknown => None,
        }
    }

    fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let delimiter = self.settings().delimiter.clone();
        if delimiter.is_empty() {
            gst::element_imp_error!(
                self,
                gst::StreamError::Format,
                ["delimiter must not be empty"]
            );
            return Err(gst::FlowError::Error);
        }

        let input = inbuf.map_readable().map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Read,
                ["Failed to map the input buffer as readable: {error}"]
            );
            gst::FlowError::Error
        })?;
        let append_delimiter = !input.as_slice().ends_with(delimiter.as_bytes());
        let output_size = if append_delimiter {
            input.len().checked_add(delimiter.len()).ok_or_else(|| {
                gst::element_imp_error!(
                    self,
                    gst::ResourceError::NoSpaceLeft,
                    ["Encoded record size exceeds the platform limit"]
                );
                gst::FlowError::Error
            })?
        } else {
            input.len()
        };

        let mut output = outbuf.map_writable().map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Write,
                ["Failed to map the output buffer as writable: {error}"]
            );
            gst::FlowError::Error
        })?;
        let output_slice = output.as_mut_slice();
        let payload_target = output_slice.get_mut(..input.len()).ok_or_else(|| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Write,
                ["Allocated output buffer is smaller than the input payload"]
            );
            gst::FlowError::Error
        })?;
        payload_target.copy_from_slice(input.as_slice());

        if append_delimiter {
            let delimiter_target =
                output_slice
                    .get_mut(input.len()..output_size)
                    .ok_or_else(|| {
                        gst::element_imp_error!(
                            self,
                            gst::ResourceError::Write,
                            ["Allocated output buffer has no room for the delimiter"]
                        );
                        gst::FlowError::Error
                    })?;
            delimiter_target.copy_from_slice(delimiter.as_bytes());
        }

        drop(output);
        outbuf.set_size(output_size);
        Ok(gst::FlowSuccess::Ok)
    }
}
