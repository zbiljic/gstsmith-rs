use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct TractTensorInference(ObjectSubclass<imp::TractTensorInference>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "tracttensorinference",
        gst::Rank::NONE,
        TractTensorInference::static_type(),
    )
}
