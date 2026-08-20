use crate::backend::{OcrError, OcrFrameResult};
use gst::glib::prelude::ToSendValue;
use gst::prelude::*;

pub(crate) fn result_structure(
    id: u64,
    generation: u64,
    pts: Option<gst::ClockTime>,
    width: u32,
    height: u32,
    latency: u64,
    result: OcrFrameResult,
) -> Result<gst::Structure, OcrError> {
    let text_bytes = result
        .lines
        .iter()
        .try_fold(0_usize, |total, line| total.checked_add(line.text.len()))
        .ok_or(OcrError::Inference)?;
    let separators = if result.lines.is_empty() {
        0
    } else {
        result
            .lines
            .len()
            .checked_sub(1)
            .ok_or(OcrError::Inference)?
    };
    let capacity = text_bytes
        .checked_add(separators)
        .ok_or(OcrError::Inference)?;
    if text_bytes > 2_097_152 || capacity > 2_097_663 || result.lines.len() > 512 {
        return Err(OcrError::Inference);
    }
    if result.lines.iter().any(|line| line.text.len() > 4_096) {
        return Err(OcrError::Inference);
    }
    let mut full_text = String::with_capacity(capacity);
    for (index, line) in result.lines.iter().enumerate() {
        if index != 0 {
            full_text.push('\n');
        }
        full_text.push_str(&line.text);
    }
    let count = u32::try_from(result.lines.len()).map_err(|_error| OcrError::Inference)?;
    let lines = result
        .lines
        .into_iter()
        .map(|line| {
            gst::Structure::builder("ocr-line")
                .field("text", line.text)
                .field("x", line.x)
                .field("y", line.y)
                .field("width", line.width)
                .field("height", line.height)
                .build()
                .to_send_value()
        })
        .collect::<gst::Array>();
    let mut structure = gst::Structure::builder("ocr-result")
        .field("request-id", id)
        .field("generation", generation)
        .field("source-width", width)
        .field("source-height", height)
        .field("latency", latency)
        .field("full-text", full_text)
        .field("line-count", count)
        .field("lines", lines)
        .build();
    if let Some(pts) = pts {
        structure.set("source-pts", pts);
    }
    Ok(structure)
}

pub(crate) fn error_structure(
    id: Option<u64>,
    generation: u64,
    error: OcrError,
    message: &'static str,
    pts: Option<gst::ClockTime>,
    width: u32,
    height: u32,
) -> gst::Structure {
    let kind = match error {
        OcrError::Input => "input",
        OcrError::Inference => "inference",
    };
    let mut structure = gst::Structure::builder("ocr-error")
        .field("generation", generation)
        .field("kind", kind)
        .field("message", message)
        .field("source-width", width)
        .field("source-height", height)
        .build();
    if let Some(id) = id {
        structure.set("request-id", id);
    }
    if let Some(pts) = pts {
        structure.set("source-pts", pts);
    }
    structure
}

pub(crate) fn post(element: &crate::ocrsanalysis::OcrsAnalysis, structure: gst::Structure) {
    let message = gst::message::Element::builder(structure)
        .src(element)
        .build();
    let _posted = element.post_message(message);
}

#[cfg(test)]
fn read_error_kind(structure: &gst::StructureRef) -> Option<String> {
    structure.get::<String>("kind").ok()
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;
    use crate::backend::OcrLine;

    fn init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            gst::init().expect("initializing GStreamer for message tests");
        });
    }

    #[test]
    fn message_result_contains_ordered_nested_lines() {
        init();
        let structure = result_structure(
            1,
            1,
            None,
            4,
            3,
            0,
            OcrFrameResult {
                lines: vec![
                    OcrLine {
                        text: "a".into(),
                        x: 0,
                        y: 0,
                        width: 1,
                        height: 1,
                    },
                    OcrLine {
                        text: "β".into(),
                        x: 1,
                        y: 1,
                        width: 2,
                        height: 2,
                    },
                ],
            },
        )
        .expect("building bounded OCR result message");
        assert_eq!(
            structure.get::<String>("full-text").ok().as_deref(),
            Some("a\nβ")
        );
        assert!(!structure.has_field("confidence"));
    }

    #[test]
    fn message_result_omits_invalid_pts_and_round_trips_nested_array() {
        init();
        let structure = result_structure(
            7,
            3,
            None,
            20,
            10,
            9,
            OcrFrameResult {
                lines: vec![OcrLine {
                    text: "line".into(),
                    x: 19,
                    y: 9,
                    width: 1,
                    height: 1,
                }],
            },
        )
        .expect("building bounded result");
        assert!(!structure.has_field("source-pts"));
        let lines = structure
            .get::<gst::Array>("lines")
            .expect("nested line array");
        let first = lines
            .iter()
            .next()
            .expect("one line")
            .get::<gst::Structure>()
            .expect("line structure");
        assert_eq!(first.name(), "ocr-line");
        assert_eq!(first.get::<u32>("x").ok(), Some(19));
        assert!(!first.has_field("confidence"));
    }

    #[test]
    fn message_errors_use_only_sanitized_local_categories() {
        init();
        for (error, kind) in [
            (OcrError::Input, "input"),
            (OcrError::Inference, "inference"),
        ] {
            let structure = error_structure(None, 1, error, "sanitized", None, 1, 1);
            assert_eq!(structure.get::<String>("kind").ok().as_deref(), Some(kind));
            assert!(!structure.has_field("request-id"));
            assert!(!structure.has_field("source-pts"));
        }
    }

    #[test]
    fn message_error_kind_reader_accepts_future_unknown_kinds() {
        init();
        let structure = gst::Structure::builder("ocr-error")
            .field("kind", "future-provider-kind")
            .build();
        assert_eq!(
            read_error_kind(&structure).as_deref(),
            Some("future-provider-kind")
        );
    }

    #[test]
    fn message_result_rejects_text_above_contract_limit() {
        init();
        let line = OcrLine {
            text: "x".repeat(2_097_153),
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        assert_eq!(
            result_structure(1, 1, None, 1, 1, 0, OcrFrameResult { lines: vec![line] }),
            Err(OcrError::Inference)
        );
    }

    #[test]
    fn message_result_enforces_per_line_and_count_limits() {
        init();
        let line = OcrLine {
            text: "x".repeat(4_097),
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        assert_eq!(
            result_structure(1, 1, None, 1, 1, 0, OcrFrameResult { lines: vec![line] }),
            Err(OcrError::Inference)
        );
        let line = OcrLine {
            text: String::new(),
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        assert_eq!(
            result_structure(
                1,
                1,
                None,
                1,
                1,
                0,
                OcrFrameResult {
                    lines: vec![line; 513]
                }
            ),
            Err(OcrError::Inference)
        );
    }

    #[test]
    fn message_result_accepts_the_maximum_success_schema() {
        init();
        let line = OcrLine {
            text: "x".repeat(4_096),
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let structure = result_structure(
            9,
            4,
            Some(gst::ClockTime::SECOND),
            1,
            1,
            2,
            OcrFrameResult {
                lines: vec![line; 512],
            },
        )
        .expect("building maximum OCR result schema");
        assert_eq!(structure.get::<u32>("line-count"), Ok(512));
        assert_eq!(
            structure
                .get::<String>("full-text")
                .expect("maximum full text")
                .len(),
            2_097_663
        );
        assert_eq!(structure.get::<u64>("request-id"), Ok(9));
        assert_eq!(
            structure.get::<gst::ClockTime>("source-pts"),
            Ok(gst::ClockTime::SECOND)
        );
    }
}
