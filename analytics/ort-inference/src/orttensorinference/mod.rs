use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct OrtTensorInference(ObjectSubclass<imp::OrtTensorInference>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "orttensorinference",
        gst::Rank::NONE,
        OrtTensorInference::static_type(),
    )
}
