use std::fs::File;
use std::io::Read;
use std::path::Path;

use ocrs::TextItem as _;

pub(crate) const MAX_AUXILIARY_BYTES: u64 = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OcrLine {
    pub(crate) text: String,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OcrFrameResult {
    pub(crate) lines: Vec<OcrLine>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrError {
    Input,
    Inference,
}

#[derive(Clone)]
struct RecognizedCharacter {
    scalar: char,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

fn recognize_candidates<T>(
    candidates: impl IntoIterator<Item = T>,
    max_lines: u32,
    mut recognize: impl FnMut(T) -> Result<Option<OcrLine>, OcrError>,
) -> Result<OcrFrameResult, OcrError> {
    let mut lines = Vec::new();
    for candidate in candidates
        .into_iter()
        .take(usize::try_from(max_lines).map_err(|_error| OcrError::Inference)?)
    {
        if let Some(line) = recognize(candidate)? {
            lines.push(line);
        }
    }
    Ok(OcrFrameResult { lines })
}

fn shape_characters(
    characters: impl IntoIterator<Item = RecognizedCharacter>,
    width: u32,
    height: u32,
    max_text_length: u32,
) -> Result<Option<OcrLine>, OcrError> {
    let mut text = String::new();
    let mut left = i32::MAX;
    let mut top = i32::MAX;
    let mut right = i32::MIN;
    let mut bottom = i32::MIN;
    let limit = usize::try_from(max_text_length).map_err(|_error| OcrError::Inference)?;
    for character in characters.into_iter().take(limit) {
        text.push(character.scalar);
        left = left.min(character.left);
        top = top.min(character.top);
        right = right.max(character.right);
        bottom = bottom.max(character.bottom);
    }
    if text.is_empty() || right <= left || bottom <= top {
        return Ok(None);
    }
    let x = u32::try_from(left.max(0))
        .map_err(|_error| OcrError::Inference)?
        .min(width);
    let y = u32::try_from(top.max(0))
        .map_err(|_error| OcrError::Inference)?
        .min(height);
    let end_x = u32::try_from(right.max(0))
        .map_err(|_error| OcrError::Inference)?
        .min(width);
    let end_y = u32::try_from(bottom.max(0))
        .map_err(|_error| OcrError::Inference)?
        .min(height);
    if end_x <= x || end_y <= y {
        return Ok(None);
    }
    Ok(Some(OcrLine {
        text,
        x,
        y,
        width: end_x.checked_sub(x).ok_or(OcrError::Inference)?,
        height: end_y.checked_sub(y).ok_or(OcrError::Inference)?,
    }))
}

pub(crate) trait OcrBackend: Send {
    fn recognize(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        max_lines: u32,
        max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError>;
}

/// Reads a trusted deployment model without exposing its path or I/O failure.
pub(crate) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, OcrError> {
    let extra = limit.checked_add(1).ok_or(OcrError::Input)?;
    let file = File::open(path).map_err(|_error| OcrError::Input)?;
    let mut bytes = Vec::new();
    file.take(extra)
        .read_to_end(&mut bytes)
        .map_err(|_error| OcrError::Input)?;
    if bytes.is_empty() || u64::try_from(bytes.len()).map_err(|_error| OcrError::Input)? > limit {
        return Err(OcrError::Input);
    }
    Ok(bytes)
}

/// Placeholder used only until valid models are loaded. It cannot produce output.
pub(crate) struct LoadedBackend(ocrs::OcrEngine);

impl OcrBackend for LoadedBackend {
    fn recognize(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        max_lines: u32,
        max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        let source = ocrs::ImageSource::from_bytes(rgb, (width, height))
            .map_err(|_error| OcrError::Input)?;
        let input = self
            .0
            .prepare_input(source)
            .map_err(|_error| OcrError::Inference)?;
        let words = self
            .0
            .detect_words(&input)
            .map_err(|_error| OcrError::Inference)?;
        let candidates = self.0.find_text_lines(&input, &words);
        recognize_candidates(candidates, max_lines, |candidate| {
            let recognized = self
                .0
                .recognize_text(&input, std::slice::from_ref(&candidate))
                .map_err(|_error| OcrError::Inference)?;
            let Some(line) = recognized.into_iter().next().ok_or(OcrError::Inference)? else {
                return Ok(None);
            };
            shape_characters(
                line.chars().iter().map(|character| RecognizedCharacter {
                    scalar: character.char,
                    left: character.rect.left(),
                    top: character.rect.top(),
                    right: character.rect.right(),
                    bottom: character.rect.bottom(),
                }),
                width,
                height,
                max_text_length,
            )
        })
    }
}

pub(crate) fn load(
    detection_model: &Path,
    recognition_model: &Path,
    max_bytes: u64,
    alphabet: Option<&Path>,
    allowed: Option<&str>,
) -> Result<Box<dyn OcrBackend>, OcrError> {
    let detection_bytes = read_bounded(detection_model, max_bytes)?;
    let recognition_bytes = read_bounded(recognition_model, max_bytes)?;
    let alphabet = alphabet.map(read_alphabet).transpose()?;
    if let Some(allowed) = allowed
        && (allowed.is_empty()
            || u64::try_from(allowed.len()).map_err(|_error| OcrError::Input)?
                > MAX_AUXILIARY_BYTES)
    {
        return Err(OcrError::Input);
    }
    let detection = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rten::Model::load(detection_bytes)
    }))
    .map_err(|_panic| OcrError::Input)?
    .map_err(|_error| OcrError::Input)?;
    let recognition = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rten::Model::load(recognition_bytes)
    }))
    .map_err(|_panic| OcrError::Input)?
    .map_err(|_error| OcrError::Input)?;
    validate_detector_model(&detection)?;
    validate_recognizer_model(&recognition)?;
    let engine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ocrs::OcrEngine::new(ocrs::OcrEngineParams {
            detection_model: Some(detection),
            recognition_model: Some(recognition),
            alphabet,
            allowed_chars: allowed.map(ToOwned::to_owned),
            ..Default::default()
        })
    }))
    .map_err(|_panic| OcrError::Input)?
    .map_err(|_error| OcrError::Input)?;
    Ok(Box::new(LoadedBackend(engine)))
}

fn read_alphabet(path: &Path) -> Result<String, OcrError> {
    let bytes = read_bounded(path, MAX_AUXILIARY_BYTES)?;
    let alphabet = String::from_utf8(bytes).map_err(|_error| OcrError::Input)?;
    let count = alphabet.chars().count();
    if count == 0 || count > 4095 {
        return Err(OcrError::Input);
    }
    Ok(alphabet)
}

fn model_input_shape(model: &rten::Model) -> Result<Vec<rten::Dimension>, OcrError> {
    let [input_id] = model.input_ids() else {
        return Err(OcrError::Input);
    };
    let [_output_id] = model.output_ids() else {
        return Err(OcrError::Input);
    };
    model
        .node_info(*input_id)
        .and_then(|input| input.shape())
        .ok_or(OcrError::Input)
}

fn batch_or_channel(dim: &rten::Dimension) -> bool {
    matches!(
        dim,
        rten::Dimension::Symbolic(_) | rten::Dimension::Fixed(1)
    )
}

fn fixed_at_most(dim: &rten::Dimension, maximum: usize) -> bool {
    matches!(dim, rten::Dimension::Fixed(value) if *value <= maximum)
}

fn fixed_spatial_product_within(
    height: &rten::Dimension,
    width: &rten::Dimension,
    maximum: usize,
) -> bool {
    let (rten::Dimension::Fixed(height), rten::Dimension::Fixed(width)) = (height, width) else {
        return false;
    };
    height
        .checked_mul(*width)
        .is_some_and(|product| product <= maximum)
}

fn symbolic_or_fixed_at_most(dim: &rten::Dimension, maximum: usize) -> bool {
    match dim {
        rten::Dimension::Symbolic(_) => true,
        rten::Dimension::Fixed(value) => *value <= maximum,
    }
}

fn validate_detector_model(model: &rten::Model) -> Result<(), OcrError> {
    validate_detector_shape(&model_input_shape(model)?)
}

fn validate_detector_shape(shape: &[rten::Dimension]) -> Result<(), OcrError> {
    let [batch, channels, height, width] = shape else {
        return Err(OcrError::Input);
    };
    if !batch_or_channel(batch)
        || !batch_or_channel(channels)
        || !fixed_at_most(height, 8192)
        || !fixed_at_most(width, 8192)
        || !fixed_spatial_product_within(height, width, 16_777_216)
    {
        return Err(OcrError::Input);
    }
    Ok(())
}

fn validate_recognizer_model(model: &rten::Model) -> Result<(), OcrError> {
    validate_recognizer_shape(&model_input_shape(model)?)
}

fn validate_recognizer_shape(shape: &[rten::Dimension]) -> Result<(), OcrError> {
    let [batch, channels, height, width] = shape else {
        return Err(OcrError::Input);
    };
    if !batch_or_channel(batch)
        || !batch_or_channel(channels)
        || !symbolic_or_fixed_at_most(height, 256)
        || !matches!(width, rten::Dimension::Symbolic(_))
    {
        return Err(OcrError::Input);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn character(
        scalar: char,
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    ) -> RecognizedCharacter {
        RecognizedCharacter {
            scalar,
            left,
            top,
            right,
            bottom,
        }
    }

    #[test]
    fn backend_empty_input_is_rejected() {
        assert_eq!(
            read_bounded(Path::new("/definitely-not-a-model"), 1),
            Err(OcrError::Input)
        );
    }

    #[test]
    fn backend_bounds_reader_rejects_empty_and_oversize_files() {
        let directory = tempfile::tempdir().expect("creating temporary model directory");
        let empty = directory.path().join("empty");
        let large = directory.path().join("large");
        std::fs::write(&empty, []).expect("writing empty fixture");
        std::fs::write(&large, [1_u8, 2, 3]).expect("writing bounded fixture");
        assert_eq!(read_bounded(&empty, 2), Err(OcrError::Input));
        assert_eq!(read_bounded(&large, 2), Err(OcrError::Input));
        assert_eq!(
            read_bounded(&large, 3).ok().as_deref(),
            Some(&[1, 2, 3][..])
        );
    }

    #[test]
    fn backend_alphabet_reader_enforces_utf8_and_scalar_bounds() {
        let directory = tempfile::tempdir().expect("creating temporary alphabet directory");
        let valid = directory.path().join("valid");
        let invalid = directory.path().join("invalid");
        let empty = directory.path().join("empty");
        std::fs::write(&valid, "aβ").expect("writing valid alphabet");
        std::fs::write(&invalid, [0xff_u8]).expect("writing invalid UTF-8 alphabet");
        std::fs::write(&empty, []).expect("writing empty alphabet");
        assert_eq!(read_alphabet(&valid).ok().as_deref(), Some("aβ"));
        assert_eq!(read_alphabet(&invalid), Err(OcrError::Input));
        assert_eq!(read_alphabet(&empty), Err(OcrError::Input));
    }

    #[test]
    fn backend_detector_shape_validation_enforces_spatial_contract() {
        let valid = vec![1.into(), 1.into(), 32.into(), 64.into()];
        assert_eq!(validate_detector_shape(&valid), Ok(()));

        let boundary = vec!["batch".into(), "channel".into(), 4096.into(), 4096.into()];
        assert_eq!(validate_detector_shape(&boundary), Ok(()));

        let oversized = vec!["batch".into(), "channel".into(), 8192.into(), 8192.into()];
        assert_eq!(validate_detector_shape(&oversized), Err(OcrError::Input));
        assert_eq!(
            validate_detector_shape(&[1.into(), 1.into(), "height".into(), 64.into()]),
            Err(OcrError::Input)
        );
        assert_eq!(
            validate_detector_shape(&[1.into(), 1.into(), 32.into()]),
            Err(OcrError::Input)
        );
    }

    #[test]
    fn backend_recognizer_shape_validation_enforces_height_and_width_contract() {
        let valid = vec![1.into(), 1.into(), 50.into(), "width".into()];
        assert_eq!(validate_recognizer_shape(&valid), Ok(()));
        assert_eq!(
            validate_recognizer_shape(&[1.into(), 1.into(), 257.into(), "width".into()]),
            Err(OcrError::Input)
        );
        assert_eq!(
            validate_recognizer_shape(&[1.into(), 1.into(), 50.into(), 2400.into()]),
            Err(OcrError::Input)
        );
        assert_eq!(
            validate_recognizer_shape(&[1.into(), 1.into(), 50.into()]),
            Err(OcrError::Input)
        );
    }

    #[test]
    fn backend_candidates_are_sequential_bounded_and_sanitized() {
        let mut long = vec![character('β', -3, -2, 3, 3), character('β', 3, 3, 20, 12)];
        for _ in 0..1_025 {
            long.push(character('β', 30, 30, 40, 40));
        }
        let candidates = vec![
            vec![character('a', -3, -2, 3, 3)],
            Vec::new(),
            long,
            vec![character('z', 0, 0, 1, 1)],
        ];
        let mut calls = Vec::new();
        let result =
            recognize_candidates(candidates.into_iter().enumerate(), 3, |(index, line)| {
                calls.push(index);
                shape_characters(line, 10, 10, 2)
            })
            .expect("processing OCR candidates");
        assert_eq!(calls, [0, 1, 2]);
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[0].text, "a");
        assert_eq!(result.lines[0].x, 0);
        assert_eq!(result.lines[0].y, 0);
        assert_eq!(result.lines[1].text, "ββ");
        assert_eq!(result.lines[1].x, 0);
        assert_eq!(result.lines[1].y, 0);
        assert_eq!(result.lines[1].width, 10);
        assert_eq!(result.lines[1].height, 10);
    }
}
