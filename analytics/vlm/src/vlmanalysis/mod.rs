use gst::{glib, prelude::*};

mod imp;

glib::wrapper! {
    pub struct VlmAnalysis(ObjectSubclass<imp::VlmAnalysis>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "vlmanalysis",
        gst::Rank::NONE,
        VlmAnalysis::static_type(),
    )
}
