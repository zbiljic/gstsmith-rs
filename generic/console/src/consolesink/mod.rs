use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct ConsoleSink(ObjectSubclass<imp::ConsoleSink>)
        @extends gst_base::BaseSink, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "consolesink",
        gst::Rank::NONE,
        ConsoleSink::static_type(),
    )
}
