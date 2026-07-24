use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;

use crate::output::text_caps;

pub struct ConsoleSrc {
    source: gst::Element,
    filter: gst::Element,
    src_pad: gst::GhostPad,
}

#[glib::object_subclass]
impl ObjectSubclass for ConsoleSrc {
    const NAME: &'static str = "GstSmithConsoleSrc";
    type Type = super::ConsoleSrc;
    type ParentType = gst::Bin;

    #[expect(
        clippy::expect_used,
        reason = "the required core elements and static source pad template are plugin invariants"
    )]
    fn with_class(class: &Self::Class) -> Self {
        let source = gst::ElementFactory::make("fdsrc")
            .name("stdin")
            .property("fd", 0_i32)
            .build()
            .expect("constructing the standard-input fdsrc");
        let filter = gst::ElementFactory::make("capsfilter")
            .name("caps")
            .property("caps", text_caps())
            .build()
            .expect("constructing the console capsfilter");
        let template = class
            .pad_template("src")
            .expect("finding the consolesrc source pad template");

        Self {
            source,
            filter,
            src_pad: gst::GhostPad::from_template(&template),
        }
    }
}

impl ObjectImpl for ConsoleSrc {
    #[expect(
        clippy::expect_used,
        reason = "the fixed internal source topology is constructed once and must be valid"
    )]
    fn constructed(&self) {
        self.parent_constructed();

        let obj = self.obj();
        obj.set_suppressed_flags(gst::ElementFlags::SINK | gst::ElementFlags::SOURCE);
        obj.set_element_flags(gst::ElementFlags::SOURCE);
        obj.add_many([&self.source, &self.filter])
            .expect("adding the consolesrc child elements");
        self.source
            .link(&self.filter)
            .expect("linking the consolesrc child elements");
        obj.add_pad(&self.src_pad)
            .expect("adding the consolesrc source pad");
        self.src_pad
            .set_target(Some(
                &self
                    .filter
                    .static_pad("src")
                    .expect("finding the capsfilter source pad"),
            ))
            .expect("targeting the consolesrc source pad");
    }
}

impl GstObjectImpl for ConsoleSrc {}

impl ElementImpl for ConsoleSrc {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Console Source",
                "Source/Generic",
                "Reads UTF-8 text or JSON buffers from standard input",
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
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &text_caps(),
            )
            .expect("constructing the consolesrc source pad template");

            vec![src]
        });

        TEMPLATES.as_ref()
    }
}

impl BinImpl for ConsoleSrc {}
