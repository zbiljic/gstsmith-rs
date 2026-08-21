use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct PrometheusTracer(ObjectSubclass<imp::PrometheusTracer>)
        @extends gst::Tracer, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Tracer::register(Some(plugin), "prometheus", PrometheusTracer::static_type())
}
