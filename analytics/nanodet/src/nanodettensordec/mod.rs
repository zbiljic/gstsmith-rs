use gst::glib;
use gst::prelude::*;

mod decode;
pub mod imp;
mod nms;

glib::wrapper! {
    pub struct NanoDetTensorDec(ObjectSubclass<imp::NanoDetTensorDec>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "nanodettensordec",
        gst::Rank::NONE,
        NanoDetTensorDec::static_type(),
    )
}
