use std::sync::LazyLock;

use gst::glib;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;

use crate::output::{Output, OutputError, text_caps};

#[derive(Default)]
pub struct ConsoleSink {
    output: Output,
}

#[glib::object_subclass]
impl ObjectSubclass for ConsoleSink {
    const NAME: &'static str = "GstSmithConsoleSink";
    type Type = super::ConsoleSink;
    type ParentType = gst_base::BaseSink;
}

impl ObjectImpl for ConsoleSink {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_sync(false);
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(Output::property_specs);

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        self.output.set_property(value, pspec);
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        self.output.property(pspec)
    }
}

impl GstObjectImpl for ConsoleSink {}

impl ElementImpl for ConsoleSink {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Console Sink",
                "Sink/Debug",
                "Writes UTF-8 buffer contents to stdout or stderr",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });

        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "constant pad names and static caps make template construction infallible"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = text_caps();
            let sink = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("constructing the consolesink pad template");

            vec![sink]
        });

        TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for ConsoleSink {
    fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
        let map = buffer.map_readable().map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Read,
                ["Failed to map the input buffer as readable: {error}"]
            );
            gst::FlowError::Error
        })?;

        self.output.write(map.as_slice()).map_err(|error| {
            match error {
                OutputError::InvalidUtf8(error) => {
                    gst::element_imp_error!(
                        self,
                        gst::StreamError::Format,
                        ["Input buffer is not valid UTF-8: {error}"]
                    );
                }
                OutputError::Write(error) => {
                    gst::element_imp_error!(
                        self,
                        gst::ResourceError::Write,
                        ["Failed to write to the console: {error}"]
                    );
                }
            }
            gst::FlowError::Error
        })?;

        Ok(gst::FlowSuccess::Ok)
    }
}
