#![expect(clippy::expect_used, reason = "test setup failures are fatal")]

use std::sync::Once;

use gst::prelude::*;

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer");
        gstocrs::plugin_register_static().expect("registering OCRs plugin");
    });
}

#[test]
fn plugin_registers_the_ocrsanalysis_factory() {
    init();
    let factory = gst::ElementFactory::find("ocrsanalysis").expect("OCRs factory is registered");
    let templates = factory.static_pad_templates();
    let sink = templates
        .iter()
        .find(|pad| pad.direction() == gst::PadDirection::Sink)
        .expect("sink pad");
    assert!(sink.caps().to_string().contains("format=(string)RGB"));
    let element = gst::ElementFactory::make("ocrsanalysis")
        .build()
        .expect("construct OCR element");
    assert!(element.find_property("backend").is_none());
    assert!(element.find_property("detection-model").is_some());
}

#[test]
fn startup_rejects_missing_local_model_paths() {
    init();
    let element = gst::ElementFactory::make("ocrsanalysis")
        .property("detection-model", "/missing-detection.rten")
        .property("recognition-model", "/missing-recognition.rten")
        .build()
        .expect("construct OCR element");
    assert!(matches!(
        element.set_state(gst::State::Paused),
        Err(gst::StateChangeError)
    ));
    let _null = element.set_state(gst::State::Null);
}

#[test]
#[ignore = "requires user-supplied compatible local OCR models and RGB data"]
fn live_ocrs_smoke() {
    init();
    let detection = std::env::var("OCRS_TEST_DETECTION_MODEL")
        .expect("OCRS_TEST_DETECTION_MODEL is required for the live smoke test");
    let recognition = std::env::var("OCRS_TEST_RECOGNITION_MODEL")
        .expect("OCRS_TEST_RECOGNITION_MODEL is required for the live smoke test");
    let image = std::env::var("OCRS_TEST_IMAGE_RGB")
        .expect("OCRS_TEST_IMAGE_RGB is required for the live smoke test");
    let width = std::env::var("OCRS_TEST_IMAGE_WIDTH")
        .expect("OCRS_TEST_IMAGE_WIDTH is required for the live smoke test")
        .parse::<i32>()
        .expect("OCRS_TEST_IMAGE_WIDTH must be an i32");
    let height = std::env::var("OCRS_TEST_IMAGE_HEIGHT")
        .expect("OCRS_TEST_IMAGE_HEIGHT is required for the live smoke test")
        .parse::<i32>()
        .expect("OCRS_TEST_IMAGE_HEIGHT must be an i32");
    assert!(
        width > 0 && height > 0,
        "live smoke dimensions must be positive"
    );
    let rgb = std::fs::read(image).expect("reading user-supplied raw RGB image");
    let expected = usize::try_from(width)
        .expect("positive width fits usize")
        .checked_mul(usize::try_from(height).expect("positive height fits usize"))
        .and_then(|pixels| pixels.checked_mul(3))
        .expect("live smoke RGB size fits usize");
    assert_eq!(rgb.len(), expected, "raw image must be tightly packed RGB");
    let element = gst::ElementFactory::make("ocrsanalysis")
        .property("detection-model", detection)
        .property("recognition-model", recognition)
        .property("analysis-interval", 0_u64)
        .build()
        .expect("constructing live OCR element");
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(
        gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", width)
            .field("height", height)
            .build(),
    );
    harness.play();
    let mut buffer = gst::Buffer::from_mut_slice(rgb);
    buffer
        .get_mut()
        .expect("new live RGB buffer is writable")
        .set_pts(gst::ClockTime::SECOND);
    assert_eq!(harness.push(buffer), Ok(gst::FlowSuccess::Ok));
    let output = harness.pull().expect("pulling live OCR passthrough buffer");
    assert_eq!(output.pts(), Some(gst::ClockTime::SECOND));
    let message = bus
        .timed_pop_filtered(
            gst::ClockTime::from_seconds(10),
            &[gst::MessageType::Element],
        )
        .expect("receiving live OCR result message");
    let gst::MessageView::Element(element_message) = message.view() else {
        panic!("live OCR message must be an element message");
    };
    let structure = element_message
        .structure()
        .expect("live OCR element message has a structure");
    assert_eq!(structure.name(), "ocr-result");
    assert_eq!(
        structure.get::<u32>("source-width"),
        Ok(width.cast_unsigned())
    );
    assert_eq!(
        structure.get::<u32>("source-height"),
        Ok(height.cast_unsigned())
    );
    assert_eq!(
        structure.get::<gst::ClockTime>("source-pts"),
        Ok(gst::ClockTime::SECOND)
    );
}
