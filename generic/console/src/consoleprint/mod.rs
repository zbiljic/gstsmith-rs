use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct ConsolePrint(ObjectSubclass<imp::ConsolePrint>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "consoleprint",
        gst::Rank::NONE,
        ConsolePrint::static_type(),
    )
}
