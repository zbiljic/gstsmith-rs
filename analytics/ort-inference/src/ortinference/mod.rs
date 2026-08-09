use gst::glib;
use gst::prelude::*;

pub mod imp;

glib::wrapper! {
    pub struct OrtInference(ObjectSubclass<imp::OrtInference>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "ortinference",
        gst::Rank::NONE,
        OrtInference::static_type(),
    )
}

#[cfg(test)]
mod tests {
    use gst::glib::prelude::*;

    use super::OrtInference;

    #[test]
    fn creates_element_with_ready_mutable_backend_properties()
    -> Result<(), Box<dyn std::error::Error>> {
        gst::init()?;
        gst::Element::register(
            None,
            "ortinference-test",
            gst::Rank::NONE,
            OrtInference::static_type(),
        )?;
        let element = gst::ElementFactory::make("ortinference-test").build()?;
        element.set_property("model-file", "model.onnx");
        element.set_property("model-info-file", "model.onnx.modelinfo");
        if element.property::<Option<String>>("model-file").as_deref() != Some("model.onnx")
            || element
                .property::<Option<String>>("model-info-file")
                .as_deref()
                != Some("model.onnx.modelinfo")
        {
            return Err(std::io::Error::other("element did not retain READY properties").into());
        }
        Ok(())
    }
}
