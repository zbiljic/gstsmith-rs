use std::sync::LazyLock;

use gst::glib;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;

use crate::output::{Output, OutputError, text_caps};

#[derive(Default)]
pub struct ConsolePrint {
    output: Output,
}

#[glib::object_subclass]
impl ObjectSubclass for ConsolePrint {
    const NAME: &'static str = "GstSmithConsolePrint";
    type Type = super::ConsolePrint;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for ConsolePrint {
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

impl GstObjectImpl for ConsolePrint {}

impl ElementImpl for ConsolePrint {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Console Print",
                "Filter/Debug",
                "Writes UTF-8 buffer contents to stdout or stderr and passes buffers through",
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
            .expect("constructing the consoleprint sink pad template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .expect("constructing the consoleprint source pad template");

            vec![sink, src]
        });

        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for ConsolePrint {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(
        &self,
        buffer: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
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
