use std::sync::{LazyLock, Mutex, MutexGuard};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;

const DEFAULT_DELIMITER: &str = "\n";
const DEFAULT_MAX_RECORD_SIZE: u32 = 65_536;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "lineparse",
        gst::DebugColorFlags::empty(),
        Some("Bounded delimiter parser"),
    )
});

#[derive(Clone)]
struct Settings {
    delimiter: String,
    max_record_size: u32,
    omit_empty: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delimiter: DEFAULT_DELIMITER.to_owned(),
            max_record_size: DEFAULT_MAX_RECORD_SIZE,
            omit_empty: false,
        }
    }
}

enum FrameAction {
    Finish { payload: Vec<u8>, consumed: u32 },
    Drop { consumed: u32 },
    NeedMore,
}

#[derive(Default)]
pub struct LineParse {
    settings: Mutex<Settings>,
}

impl LineParse {
    fn settings(&self) -> MutexGuard<'_, Settings> {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn format_error(&self, detail: &str) -> gst::FlowError {
        gst::element_imp_error!(
            self,
            gst::StreamError::Format,
            ["Invalid delimiter-framed stream: {detail}"]
        );
        gst::FlowError::Error
    }

    fn checked_lookahead(settings: &Settings) -> Result<u32, gst::ErrorMessage> {
        let delimiter_len =
            u32::try_from(settings.delimiter.len()).map_err(|conversion_error| {
                gst::error_msg!(
                    gst::CoreError::StateChange,
                    ["Delimiter is too large to configure bounded parsing: {conversion_error}"]
                )
            })?;

        settings
            .max_record_size
            .checked_add(delimiter_len)
            .ok_or_else(|| {
                gst::error_msg!(
                    gst::CoreError::StateChange,
                    ["max-record-size plus delimiter size exceeds the parser limit"]
                )
            })
    }

    fn frame_action(
        &self,
        frame: &gst_base::BaseParseFrame,
        settings: &Settings,
        max_record_size: usize,
        lookahead: usize,
        draining: bool,
    ) -> Result<FrameAction, gst::FlowError> {
        let input = frame
            .buffer()
            .ok_or_else(|| self.format_error("parser received a frame without a buffer"))?;
        let map = input.map_readable().map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Read,
                ["Failed to map the accumulated input buffer as readable: {error}"]
            );
            gst::FlowError::Error
        })?;
        let data = map.as_slice();

        if let Some(record_len) = memchr::memmem::find(data, settings.delimiter.as_bytes()) {
            if record_len > max_record_size {
                return Err(self.format_error("record exceeds max-record-size"));
            }
            let consumed = record_len
                .checked_add(settings.delimiter.len())
                .and_then(|size| u32::try_from(size).ok())
                .ok_or_else(|| self.format_error("record size is not representable"))?;

            if record_len == 0 && settings.omit_empty {
                return Ok(FrameAction::Drop { consumed });
            }

            let payload = data
                .get(..record_len)
                .ok_or_else(|| self.format_error("invalid record boundary"))?
                .to_vec();
            return Ok(FrameAction::Finish { payload, consumed });
        }

        if draining {
            if data.is_empty() {
                return Ok(FrameAction::NeedMore);
            }
            if data.len() > max_record_size {
                return Err(self.format_error("unterminated record exceeds max-record-size"));
            }
            let consumed = u32::try_from(data.len()).map_err(|conversion_error| {
                self.format_error(&format!(
                    "record size is not representable: {conversion_error}"
                ))
            })?;
            return Ok(FrameAction::Finish {
                payload: data.to_vec(),
                consumed,
            });
        }

        if data.len() >= lookahead {
            return Err(
                self.format_error("buffered data cannot contain a record within max-record-size")
            );
        }

        Ok(FrameAction::NeedMore)
    }
}

#[glib::object_subclass]
impl ObjectSubclass for LineParse {
    const NAME: &'static str = "GstSmithLineParse";
    type Type = super::LineParse;
    type ParentType = gst_base::BaseParse;
}

impl ObjectImpl for LineParse {
    fn constructed(&self) {
        self.parent_constructed();

        let parser = self.obj().downgrade();
        let _probe_id = self.obj().sink_pad().add_probe(
            gst::PadProbeType::EVENT_DOWNSTREAM | gst::PadProbeType::EVENT_FLUSH,
            move |_pad, info| {
                if let Some(gst::PadProbeData::Event(event)) = info.data.as_ref()
                    && matches!(
                        event.view(),
                        gst::EventView::FlushStop(_) | gst::EventView::StreamStart(_)
                    )
                    && let Some(parser) = parser.upgrade()
                {
                    parser.set_min_frame_size(1);
                }
                gst::PadProbeReturn::Ok
            },
        );
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("delimiter")
                    .nick("Delimiter")
                    .blurb("Non-empty delimiter removed from each output record")
                    .default_value(Some(DEFAULT_DELIMITER))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-record-size")
                    .nick("Maximum Record Size")
                    .blurb("Maximum record payload size in bytes, excluding the delimiter")
                    .minimum(1)
                    .default_value(DEFAULT_MAX_RECORD_SIZE)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("omit-empty")
                    .nick("Omit Empty Records")
                    .blurb("Drop empty records between adjacent delimiters")
                    .default_value(false)
                    .mutable_ready()
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings();
        match pspec.name() {
            "delimiter" => {
                if let Ok(delimiter) = value.get::<String>() {
                    settings.delimiter = delimiter;
                }
            }
            "max-record-size" => {
                if let Ok(max_record_size) = value.get::<u32>() {
                    settings.max_record_size = max_record_size;
                }
            }
            "omit-empty" => {
                if let Ok(omit_empty) = value.get::<bool>() {
                    settings.omit_empty = omit_empty;
                }
            }
            _ => gst::warning!(CAT, imp = self, "Unknown property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings();
        match pspec.name() {
            "delimiter" => settings.delimiter.to_value(),
            "max-record-size" => settings.max_record_size.to_value(),
            "omit-empty" => settings.omit_empty.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for LineParse {}

impl ElementImpl for LineParse {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Delimiter Parser",
                "Codec/Parser/Text",
                "Frames a byte stream into bounded delimiter-separated records",
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
            .expect("constructing the lineparse sink pad template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("constructing the lineparse source pad template");

            vec![sink, src]
        });

        TEMPLATES.as_ref()
    }
}

impl BaseParseImpl for LineParse {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        if settings.delimiter.is_empty() {
            return Err(gst::error_msg!(
                gst::CoreError::StateChange,
                ["delimiter must not be empty"]
            ));
        }

        let _lookahead = Self::checked_lookahead(&settings)?;
        self.obj().set_min_frame_size(1);
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        self.obj().set_min_frame_size(1);
        Ok(())
    }

    fn set_sink_caps(&self, caps: &gst::Caps) -> Result<(), gst::LoggableError> {
        if caps.is_fixed() && !self.obj().src_pad().push_event(gst::event::Caps::new(caps)) {
            return Err(gst::loggable_error!(
                CAT,
                "Failed to propagate fixed sink caps to the source pad"
            ));
        }
        Ok(())
    }

    fn handle_frame(
        &self,
        mut frame: gst_base::BaseParseFrame,
    ) -> Result<(gst::FlowSuccess, u32), gst::FlowError> {
        let settings = self.settings().clone();
        if settings.delimiter.is_empty() {
            return Err(self.format_error("delimiter must not be empty"));
        }

        let max_record_size =
            usize::try_from(settings.max_record_size).map_err(|conversion_error| {
                self.format_error(&format!(
                    "max-record-size is not representable: {conversion_error}"
                ))
            })?;
        let lookahead_u32 = Self::checked_lookahead(&settings).map_err(|configuration_error| {
            self.format_error(&format!(
                "configured record bound is not representable: {configuration_error}"
            ))
        })?;
        let lookahead = usize::try_from(lookahead_u32).map_err(|conversion_error| {
            self.format_error(&format!(
                "configured look-ahead is not representable: {conversion_error}"
            ))
        })?;
        let draining = self.obj().is_draining();
        let action = self.frame_action(&frame, &settings, max_record_size, lookahead, draining)?;

        match action {
            FrameAction::Finish { payload, consumed } => {
                let mut output = gst::Buffer::from_mut_slice(payload);
                let input = frame
                    .buffer()
                    .ok_or_else(|| self.format_error("parser received a frame without a buffer"))?;
                input
                    .copy_into(output.make_mut(), gst::BufferCopyFlags::FLAGS, ..)
                    .map_err(|error| {
                        gst::element_imp_error!(
                            self,
                            gst::ResourceError::Write,
                            ["Failed to preserve input buffer flags: {error}"]
                        );
                        gst::FlowError::Error
                    })?;
                frame.set_output_buffer(output);
                self.obj().set_min_frame_size(1);
                self.obj().finish_frame(frame, consumed)?;
                Ok((gst::FlowSuccess::Ok, 0))
            }
            FrameAction::Drop { consumed } => {
                frame.set_flags(gst_base::BaseParseFrameFlags::DROP);
                self.obj().set_min_frame_size(1);
                self.obj().finish_frame(frame, consumed)?;
                Ok((gst::FlowSuccess::Ok, 0))
            }
            FrameAction::NeedMore => {
                if !draining {
                    let next_size = frame
                        .buffer()
                        .and_then(|buffer| buffer.size().checked_add(1))
                        .and_then(|size| u32::try_from(size).ok())
                        .map_or(lookahead_u32, |size| size.min(lookahead_u32));
                    self.obj().set_min_frame_size(next_size);
                }
                Ok((gst::FlowSuccess::Ok, 0))
            }
        }
    }
}
