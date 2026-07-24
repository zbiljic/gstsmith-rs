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

#[derive(Clone, Copy, Debug)]
struct Settings {
    stream: ConsoleStream,
    ensure_newline: bool,
}

impl Default for Settings {
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
pub(crate) struct Output {
    settings: Mutex<Settings>,
}

impl Output {
    pub(crate) fn property_specs() -> Vec<glib::ParamSpec> {
        vec![
            glib::ParamSpecEnum::builder::<ConsoleStream>("stream")
                .nick("Stream")
                .blurb("Standard stream to write to")
                .default_value(ConsoleStream::Stdout)
                .build(),
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
        let record =
            prepare_record(payload, settings.ensure_newline).map_err(OutputError::InvalidUtf8)?;

        match settings.stream {
            ConsoleStream::Stdout => {
                let mut stream = io::stdout().lock();
                write_record(&mut stream, &record).map_err(OutputError::Write)
            }
            ConsoleStream::Stderr => {
                let mut stream = io::stderr().lock();
                write_record(&mut stream, &record).map_err(OutputError::Write)
            }
        }
    }

    fn lock_settings(&self) -> MutexGuard<'_, Settings> {
        self.settings.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn prepare_record(payload: &[u8], ensure_newline: bool) -> Result<Cow<'_, [u8]>, Utf8Error> {
    let _text = std::str::from_utf8(payload)?;

    if !ensure_newline || payload.ends_with(b"\n") {
        return Ok(Cow::Borrowed(payload));
    }

    let mut record = Vec::with_capacity(payload.len().saturating_add(1));
    record.extend_from_slice(payload);
    record.push(b'\n');
    Ok(Cow::Owned(record))
}

fn write_record(writer: &mut impl Write, record: &[u8]) -> io::Result<()> {
    writer.write_all(record)?;
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
        let record = prepare_record(b"hello", false).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello");
        assert!(matches!(record, Cow::Borrowed(_)));
    }

    #[test]
    fn adds_one_missing_newline() {
        let record = prepare_record(b"hello", true).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello\n");
        assert!(matches!(record, Cow::Owned(_)));
    }

    #[test]
    fn preserves_existing_newline() {
        let record = prepare_record(b"hello\n", true).expect("valid UTF-8");

        assert_eq!(record.as_ref(), b"hello\n");
        assert!(matches!(record, Cow::Borrowed(_)));
    }

    #[test]
    fn rejects_invalid_utf8() {
        let _error = prepare_record(&[0xff], true).expect_err("invalid UTF-8 must be rejected");
    }

    #[test]
    fn writes_the_prepared_record() {
        let mut output = Vec::new();

        write_record(&mut output, b"hello\n").expect("write to memory");

        assert_eq!(output, b"hello\n");
    }
}
