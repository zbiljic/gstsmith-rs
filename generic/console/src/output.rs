use std::borrow::Cow;
use std::io::{self, Write};
use std::str::Utf8Error;
use std::sync::{Mutex, MutexGuard, PoisonError};

use gst::glib;
use gst::glib::prelude::*;

/// Standard stream used by the console elements.
#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithConsoleStream")]
pub enum ConsoleStream {
    /// Write records to the process's standard output stream.
    #[default]
    #[enum_value(name = "Standard output", nick = "stdout")]
    Stdout = 0,

    /// Write records to the process's standard error stream.
    #[enum_value(name = "Standard error", nick = "stderr")]
    Stderr = 1,
}

#[derive(Clone, Copy, Debug, Default)]
struct RawSettings {
    stream: ConsoleStream,
}

#[derive(Clone, Copy, Debug)]
struct TextSettings {
    stream: ConsoleStream,
    ensure_newline: bool,
}

impl Default for TextSettings {
    fn default() -> Self {
        Self {
            stream: ConsoleStream::Stdout,
            ensure_newline: true,
        }
    }
}

#[derive(Debug)]
pub(crate) enum OutputError {
    InvalidUtf8(Utf8Error),
    Write(io::Error),
}

#[derive(Debug, Default)]
pub(crate) struct RawOutput {
    settings: Mutex<RawSettings>,
}

impl RawOutput {
    pub(crate) fn property_specs() -> Vec<glib::ParamSpec> {
        vec![stream_property_spec()]
    }

    pub(crate) fn set_property(&self, value: &glib::Value, pspec: &glib::ParamSpec) {
        if pspec.name() == "stream"
            && let Ok(stream) = value.get::<ConsoleStream>()
        {
            self.lock_settings().stream = stream;
        }
    }

    pub(crate) fn property(&self, pspec: &glib::ParamSpec) -> glib::Value {
        if pspec.name() == "stream" {
            return self.lock_settings().stream.to_value();
        }

        pspec.default_value().clone()
    }

    pub(crate) fn write(&self, payload: &[u8]) -> io::Result<()> {
        let stream = self.lock_settings().stream;

        write_to_stream(stream, payload)
    }

    fn lock_settings(&self) -> MutexGuard<'_, RawSettings> {
        self.settings.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Debug, Default)]
pub(crate) struct TextOutput {
    settings: Mutex<TextSettings>,
}

impl TextOutput {
    pub(crate) fn property_specs() -> Vec<glib::ParamSpec> {
        vec![
            stream_property_spec(),
            glib::ParamSpecBoolean::builder("ensure-newline")
                .nick("Ensure newline")
                .blurb("Append a newline when the buffer does not already end with one")
                .default_value(true)
                .build(),
        ]
    }

    pub(crate) fn set_property(&self, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.lock_settings();

        match pspec.name() {
            "stream" => {
                if let Ok(stream) = value.get::<ConsoleStream>() {
                    settings.stream = stream;
                }
            }
            "ensure-newline" => {
                if let Ok(ensure_newline) = value.get::<bool>() {
                    settings.ensure_newline = ensure_newline;
                }
            }
            _ => {}
        }
    }

    pub(crate) fn property(&self, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.lock_settings();

        match pspec.name() {
            "stream" => settings.stream.to_value(),
            "ensure-newline" => settings.ensure_newline.to_value(),
            _ => pspec.default_value().clone(),
        }
    }

    pub(crate) fn write(&self, payload: &[u8]) -> Result<(), OutputError> {
        let settings = *self.lock_settings();
        let record = prepare_text_record(payload, settings.ensure_newline)
            .map_err(OutputError::InvalidUtf8)?;

        write_to_stream(settings.stream, &record).map_err(OutputError::Write)
    }

    fn lock_settings(&self) -> MutexGuard<'_, TextSettings> {
        self.settings.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn stream_property_spec() -> glib::ParamSpec {
    glib::ParamSpecEnum::builder::<ConsoleStream>("stream")
        .nick("Stream")
        .blurb("Standard stream to write to")
        .default_value(ConsoleStream::Stdout)
        .build()
}

fn prepare_text_record(payload: &[u8], ensure_newline: bool) -> Result<Cow<'_, [u8]>, Utf8Error> {
    let _text = std::str::from_utf8(payload)?;

    if !ensure_newline || payload.ends_with(b"\n") {
        return Ok(Cow::Borrowed(payload));
    }

    let mut record = Vec::with_capacity(payload.len().saturating_add(1));
    record.extend_from_slice(payload);
    record.push(b'\n');
    Ok(Cow::Owned(record))
}

fn write_to_stream(stream: ConsoleStream, payload: &[u8]) -> io::Result<()> {
    match stream {
        ConsoleStream::Stdout => {
            let mut stdout = io::stdout().lock();
            write_bytes(&mut stdout, payload)
        }
        ConsoleStream::Stderr => {
            let mut stderr = io::stderr().lock();
            write_bytes(&mut stderr, payload)
        }
    }
}

fn write_bytes(writer: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    writer.write_all(payload)?;
    writer.flush()
}

pub(crate) fn text_caps() -> gst::Caps {
    gst::Caps::builder_full()
        .structure(
            gst::Structure::builder("text/x-raw")
                .field("format", "utf8")
                .build(),
        )
        .structure(gst::Structure::builder("application/json").build())
        .structure(gst::Structure::builder("application/x-json").build())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_payload_unchanged_when_newline_is_disabled() {
        let record = prepare_text_record(b"hello", false).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello");
        assert!(matches!(record, Cow::Borrowed(_)));
    }

    #[test]
    fn adds_one_missing_newline() {
        let record = prepare_text_record(b"hello", true).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello\n");
        assert!(matches!(record, Cow::Owned(_)));
    }

    #[test]
    fn preserves_existing_newline() {
        let record = prepare_text_record(b"hello\n", true).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello\n");
        assert!(matches!(record, Cow::Borrowed(_)));
    }

    #[test]
    fn rejects_invalid_utf8() {
        let _error =
            prepare_text_record(&[0xff], true).expect_err("invalid UTF-8 must be rejected");
    }

    #[test]
    fn raw_writer_preserves_arbitrary_bytes_exactly() {
        let mut output = Vec::new();

        write_bytes(&mut output, &[0x00, 0xff, b'\n']).expect("write to memory");

        assert_eq!(output, &[0x00, 0xff, b'\n']);
    }
}
