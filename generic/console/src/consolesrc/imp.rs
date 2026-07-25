use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;

pub struct ConsoleSrc {
    source: gst::Element,
    src_pad: gst::GhostPad,
}

#[glib::object_subclass]
impl ObjectSubclass for ConsoleSrc {
    const NAME: &'static str = "GstSmithConsoleSrc";
    type Type = super::ConsoleSrc;
    type ParentType = gst::Bin;

    #[expect(
        clippy::expect_used,
        reason = "the required core source element and static source pad template are plugin invariants"
    )]
    fn with_class(class: &Self::Class) -> Self {
        let source = gst::ElementFactory::make("fdsrc")
            .name("stdin")
            .property("fd", 0_i32)
            .build()
            .expect("constructing the standard-input fdsrc");
        let template = class
            .pad_template("src")
            .expect("finding the consolesrc source pad template");

        Self {
            source,
            src_pad: gst::GhostPad::from_template(&template),
        }
    }
}

impl ObjectImpl for ConsoleSrc {
    #[expect(
        clippy::expect_used,
        reason = "the fixed fdsrc source topology is constructed once and must be valid"
    )]
    fn constructed(&self) {
        self.parent_constructed();

        let obj = self.obj();
        obj.set_suppressed_flags(gst::ElementFlags::SINK | gst::ElementFlags::SOURCE);
        obj.set_element_flags(gst::ElementFlags::SOURCE);
        obj.add(&self.source)
            .expect("adding the consolesrc fdsrc child element");
        obj.add_pad(&self.src_pad)
            .expect("adding the consolesrc source pad");
        self.src_pad
            .set_target(Some(
                &self
                    .source
                    .static_pad("src")
                    .expect("finding the fdsrc source pad"),
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
                "Reads bytes from standard input",
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
                &gst::Caps::new_any(),
            )
            .expect("constructing the consolesrc source pad template");

            vec![src]
        });

        TEMPLATES.as_ref()
    }
}

impl BinImpl for ConsoleSrc {}
