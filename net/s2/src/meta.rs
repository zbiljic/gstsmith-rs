use gst::glib;
use gst::prelude::*;
use s2_sdk::types::{Header, SequencedRecord};

pub const META_NAME: &str = "GstS2RecordMeta";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderValue {
    pub name: Vec<u8>,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Envelope {
    pub basin: String,
    pub stream: String,
    pub seq_num: u64,
    pub timestamp: u64,
    pub is_command: bool,
    pub headers: Vec<HeaderValue>,
}

pub fn register() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        if !gst::meta::CustomMeta::is_registered(META_NAME) {
            gst::meta::CustomMeta::register_simple(META_NAME);
        }
    });
}

pub fn attach_record(
    buffer: &mut gst::BufferRef,
    basin: &str,
    stream: &str,
    record: &SequencedRecord,
) -> Result<(), String> {
    let headers = record
        .headers
        .iter()
        .map(|header| HeaderValue {
            name: header.name.to_vec(),
            value: header.value.to_vec(),
        })
        .collect();
    attach(
        buffer,
        &Envelope {
            basin: basin.to_owned(),
            stream: stream.to_owned(),
            seq_num: record.seq_num,
            timestamp: record.timestamp,
            is_command: record.is_command_record(),
            headers,
        },
    )
}

pub fn attach(buffer: &mut gst::BufferRef, envelope: &Envelope) -> Result<(), String> {
    let mut meta = gst::meta::CustomMeta::add(buffer, META_NAME)
        .map_err(|error| format!("failed to attach S2 record metadata: {error}"))?;
    let structure = meta.mut_structure();
    structure.set("basin", envelope.basin.as_str());
    structure.set("stream", envelope.stream.as_str());
    structure.set("seq-num", envelope.seq_num);
    structure.set("timestamp", envelope.timestamp);
    structure.set("is-command", envelope.is_command);
    structure.set("headers", headers_to_array(&envelope.headers));
    Ok(())
}

pub fn read(buffer: &gst::BufferRef) -> Result<Envelope, String> {
    let meta = gst::meta::CustomMeta::from_buffer(buffer, META_NAME)
        .map_err(|_missing_error| "buffer has no GstS2RecordMeta".to_owned())?;
    let structure = meta.structure();
    let basin = required::<String>(structure, "basin")?;
    let stream = required::<String>(structure, "stream")?;
    let seq_num = required::<u64>(structure, "seq-num")?;
    let timestamp = required::<u64>(structure, "timestamp")?;
    let is_command = required::<bool>(structure, "is-command")?;
    let array = required::<gst::Array>(structure, "headers")?;
    let headers = headers_from_array(&array)?;
    let represented_as_command =
        headers.len() == 1 && headers.first().is_some_and(|header| header.name.is_empty());
    if is_command != represented_as_command {
        return Err(
            "GstS2RecordMeta is-command disagrees with its command header representation"
                .to_owned(),
        );
    }
    Ok(Envelope {
        basin,
        stream,
        seq_num,
        timestamp,
        is_command,
        headers,
    })
}

pub fn is_present(buffer: &gst::BufferRef) -> bool {
    gst::meta::CustomMeta::from_buffer(buffer, META_NAME).is_ok()
}

pub fn regular_headers(envelope: &Envelope) -> Result<Vec<Header>, String> {
    if envelope.is_command || envelope.headers.iter().any(|header| header.name.is_empty()) {
        return Err("S2 command record metadata cannot be appended by s2sink".to_owned());
    }
    Ok(envelope
        .headers
        .iter()
        .map(|header| Header::new(header.name.clone(), header.value.clone()))
        .collect())
}

fn required<'a, T>(structure: &'a gst::StructureRef, field: &str) -> Result<T, String>
where
    T: glib::value::FromValue<'a>,
{
    structure
        .get::<T>(field)
        .map_err(|_field_error| format!("GstS2RecordMeta {field} is missing or invalid"))
}

fn headers_to_array(headers: &[HeaderValue]) -> gst::Array {
    headers
        .iter()
        .map(|header| {
            gst::Structure::builder("s2-header")
                .field("name", glib::Bytes::from_owned(header.name.clone()))
                .field("value", glib::Bytes::from_owned(header.value.clone()))
                .build()
                .to_send_value()
        })
        .collect()
}

fn headers_from_array(array: &gst::Array) -> Result<Vec<HeaderValue>, String> {
    array
        .iter()
        .map(|value| {
            let structure = value
                .get::<gst::Structure>()
                .map_err(|_type_error| "S2 header entry is not a structure".to_owned())?;
            if structure.name() != "s2-header" {
                return Err("S2 header structure must be named s2-header".to_owned());
            }
            let name = structure
                .get::<glib::Bytes>("name")
                .map_err(|_field_error| "S2 header name is missing or is not bytes".to_owned())?;
            let value = structure
                .get::<glib::Bytes>("value")
                .map_err(|_field_error| "S2 header value is missing or is not bytes".to_owned())?;
            Ok(HeaderValue {
                name: name.as_ref().to_vec(),
                value: value.as_ref().to_vec(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(is_command: bool, headers: Vec<HeaderValue>) -> Envelope {
        Envelope {
            basin: "test-basin".to_owned(),
            stream: "test-stream".to_owned(),
            seq_num: 0,
            timestamp: 0,
            is_command,
            headers,
        }
    }

    #[test]
    fn inconsistent_command_metadata_is_rejected() {
        gst::init().expect("initializing GStreamer");
        register();
        for malformed in [
            envelope(
                false,
                vec![HeaderValue {
                    name: Vec::new(),
                    value: b"fence".to_vec(),
                }],
            ),
            envelope(
                true,
                vec![HeaderValue {
                    name: b"ordinary".to_vec(),
                    value: Vec::new(),
                }],
            ),
        ] {
            let mut buffer = gst::Buffer::new();
            attach(
                buffer.get_mut().expect("new buffer is writable"),
                &malformed,
            )
            .expect("attaching deliberately malformed metadata");
            read(&buffer).expect_err("command marker disagreement must be rejected");
        }
    }

    #[test]
    fn regular_header_conversion_rejects_command_records() {
        let command = envelope(
            true,
            vec![HeaderValue {
                name: Vec::new(),
                value: b"trim".to_vec(),
            }],
        );
        regular_headers(&command).expect_err("command metadata cannot be appended");
    }
}
