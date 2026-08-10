use gst::glib;
use gst::prelude::*;

pub mod imp;

glib::wrapper! {
    pub struct TractInference(ObjectSubclass<imp::TractInference>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "tractinference",
        gst::Rank::NONE,
        TractInference::static_type(),
    )
}

#[cfg(test)]
mod tests {
    use gst::glib::prelude::*;

    use super::{TractInference, imp};

    #[test]
    fn creates_an_element_and_keeps_ready_properties_mutable()
    -> Result<(), Box<dyn std::error::Error>> {
        gst::init()?;
        gst::Element::register(
            None,
            "tractinference-test",
            gst::Rank::NONE,
            TractInference::static_type(),
        )?;
        let element = gst::ElementFactory::make("tractinference-test").build()?;
        element.set_property("model-file", "model.onnx");
        element.set_property("model-info-file", "model.onnx.modelinfo");
        element.set_property("model-channel-order", imp::ModelChannelOrder::Bgr);
        let model_file = element.property::<Option<String>>("model-file");
        let info_file = element.property::<Option<String>>("model-info-file");
        if model_file.as_deref() != Some("model.onnx")
            || info_file.as_deref() != Some("model.onnx.modelinfo")
            || element
                .property_value("model-channel-order")
                .get::<imp::ModelChannelOrder>()?
                != imp::ModelChannelOrder::Bgr
        {
            return Err(std::io::Error::other("element did not retain READY properties").into());
        }
        Ok(())
    }
}
