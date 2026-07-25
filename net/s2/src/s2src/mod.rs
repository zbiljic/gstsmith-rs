use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct S2Src(ObjectSubclass<imp::S2Src>)
        @extends gst_base::PushSrc, gst_base::BaseSrc, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(Some(plugin), "s2src", gst::Rank::NONE, S2Src::static_type())
}
