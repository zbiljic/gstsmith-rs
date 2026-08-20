use gst::glib;
use gst::prelude::*;

mod imp;
#[cfg(test)]
mod tests;

glib::wrapper! {
    pub struct OcrsAnalysis(ObjectSubclass<imp::OcrsAnalysis>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "ocrsanalysis",
        gst::Rank::NONE,
        OcrsAnalysis::static_type(),
    )
}
