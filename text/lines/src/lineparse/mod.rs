use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct LineParse(ObjectSubclass<imp::LineParse>)
        @extends gst_base::BaseParse, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "lineparse",
        gst::Rank::NONE,
        LineParse::static_type(),
    )
}
