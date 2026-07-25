use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct NatsSink(ObjectSubclass<imp::NatsSink>)
        @extends gst_base::BaseSink, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "natssink",
        gst::Rank::NONE,
        NatsSink::static_type(),
    )
}
