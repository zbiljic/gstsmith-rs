use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct S2Sink(ObjectSubclass<imp::S2Sink>)
        @extends gst_base::BaseSink, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "s2sink",
        gst::Rank::NONE,
        S2Sink::static_type(),
    )
}
