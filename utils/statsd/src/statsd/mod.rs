use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct StatsdTracer(ObjectSubclass<imp::StatsdTracer>)
        @extends gst::Tracer, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Tracer::register(Some(plugin), "statsd", StatsdTracer::static_type())
}
