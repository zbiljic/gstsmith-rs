use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct ConsoleSrc(ObjectSubclass<imp::ConsoleSrc>)
        @extends gst::Bin, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "consolesrc",
        gst::Rank::NONE,
        ConsoleSrc::static_type(),
    )
}
