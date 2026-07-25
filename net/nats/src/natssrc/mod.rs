use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct NatsSrc(ObjectSubclass<imp::NatsSrc>)
        @extends gst_base::PushSrc, gst_base::BaseSrc, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "natssrc",
        gst::Rank::NONE,
        NatsSrc::static_type(),
    )
}
