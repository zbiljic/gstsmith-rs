use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct LineEnc(ObjectSubclass<imp::LineEnc>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "lineenc",
        gst::Rank::NONE,
        LineEnc::static_type(),
    )
}
