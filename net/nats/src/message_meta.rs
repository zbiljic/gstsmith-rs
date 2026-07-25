use std::str::FromStr;

use gst::prelude::*;

pub const META_NAME: &str = "GstNatsMessageMeta";

#[derive(Clone, Debug)]
pub struct Envelope {
    pub subject: String,
    pub reply_subject: Option<String>,
    pub headers: Option<async_nats::HeaderMap>,
}

pub fn register() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        if !gst::meta::CustomMeta::is_registered(META_NAME) {
            gst::meta::CustomMeta::register_simple(META_NAME);
        }
    });
}

pub fn attach(
    buffer: &mut gst::BufferRef,
    subject: &str,
    reply_subject: Option<&str>,
    headers: Option<&async_nats::HeaderMap>,
) -> Result<(), String> {
    validate_subject(subject)?;
    if let Some(reply) = reply_subject {
        validate_subject(reply)?;
    }

    let mut meta = gst::meta::CustomMeta::add(buffer, META_NAME)
        .map_err(|error| format!("failed to attach NATS message metadata: {error}"))?;
    let structure = meta.mut_structure();
    structure.set("subject", subject);
    if let Some(reply) = reply_subject.filter(|reply| !reply.is_empty()) {
        structure.set("reply-subject", reply);
    }
    if let Some(headers) = headers.filter(|headers| !headers.is_empty()) {
        structure.set("headers", headers_to_array(headers));
    }
    Ok(())
}

pub fn read(buffer: &gst::BufferRef) -> Result<Envelope, String> {
    let meta = gst::meta::CustomMeta::from_buffer(buffer, META_NAME)
        .map_err(|_error| "buffer has no GstNatsMessageMeta".to_owned())?;
    let structure = meta.structure();
    let subject = structure
        .get::<String>("subject")
        .map_err(|_error| "GstNatsMessageMeta subject is missing or is not a string".to_owned())?;
    validate_subject(&subject)?;

    let reply_subject = match structure.get_optional::<String>("reply-subject") {
        Ok(Some(reply)) => {
            if reply.is_empty() {
                None
            } else {
                validate_subject(&reply)?;
                Some(reply)
            }
        }
        Ok(None) => None,
        Err(_) => {
            return Err("GstNatsMessageMeta reply-subject is not a string".to_owned());
        }
    };

    let headers = match structure.get_optional::<gst::Array>("headers") {
        Ok(Some(array)) => Some(headers_from_array(&array)?),
        Ok(None) => None,
        Err(_) => return Err("GstNatsMessageMeta headers is not an array".to_owned()),
    };

    Ok(Envelope {
        subject,
        reply_subject,
        headers,
    })
}

pub fn is_present(buffer: &gst::BufferRef) -> bool {
    gst::meta::CustomMeta::from_buffer(buffer, META_NAME).is_ok()
}

fn validate_subject(subject: &str) -> Result<(), String> {
    if subject.is_empty()
        || subject
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte == b'\0')
    {
        Err("NATS subject must be non-empty and contain no whitespace".to_owned())
    } else {
        Ok(())
    }
}

fn headers_to_array(headers: &async_nats::HeaderMap) -> gst::Array {
    let values = headers.iter().flat_map(|(name, values)| {
        values.iter().map(move |value| {
            gst::Structure::builder("nats-header")
                .field("name", name.to_string())
                .field("value", value.as_str())
                .build()
                .to_send_value()
        })
    });
    values.collect::<gst::Array>()
}

pub fn headers_from_array(array: &gst::Array) -> Result<async_nats::HeaderMap, String> {
    let mut headers = async_nats::HeaderMap::new();
    for value in array.iter() {
        let structure = value
            .get::<gst::Structure>()
            .map_err(|_error| "NATS header entry is not a structure".to_owned())?;
        if structure.name() != "nats-header" {
            return Err("NATS header structure must be named nats-header".to_owned());
        }
        let name = structure
            .get::<String>("name")
            .map_err(|_error| "NATS header name is missing or invalid".to_owned())?;
        let value = structure
            .get::<String>("value")
            .map_err(|_error| "NATS header value is missing or invalid".to_owned())?;
        let name = async_nats::HeaderName::from_str(&name)
            .map_err(|_error| "NATS header name is invalid".to_owned())?;
        let value = async_nats::HeaderValue::from_str(&value)
            .map_err(|_error| "NATS header value is invalid".to_owned())?;
        headers.append(name, value);
    }
    Ok(headers)
}

pub fn merge_headers(
    fixed: Option<&async_nats::HeaderMap>,
    message: Option<&async_nats::HeaderMap>,
) -> Option<async_nats::HeaderMap> {
    let mut merged = async_nats::HeaderMap::new();
    for headers in [fixed, message].into_iter().flatten() {
        for (name, values) in headers.iter() {
            for value in values {
                merged.append(name.clone(), value.clone());
            }
        }
    }
    (!merged.is_empty()).then_some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        gst::init().expect("initializing GStreamer");
        register();
    }

    #[test]
    fn message_meta_round_trip_and_copy() {
        init();
        let mut headers = async_nats::HeaderMap::new();
        headers.append("X-Test", "one");
        headers.append("X-Test", "two");
        let mut buffer = gst::Buffer::new();
        attach(
            buffer.get_mut().expect("new buffer is writable"),
            "actual.subject",
            Some("reply.subject"),
            Some(&headers),
        )
        .expect("metadata attaches");

        let copied = buffer.copy();
        let envelope = read(&copied).expect("metadata reads from copied buffer");
        assert_eq!(envelope.subject, "actual.subject");
        assert_eq!(envelope.reply_subject.as_deref(), Some("reply.subject"));
        assert_eq!(
            envelope
                .headers
                .expect("headers exist")
                .get_all("X-Test")
                .count(),
            2
        );
    }

    #[test]
    fn message_meta_without_headers_round_trips() {
        init();
        let mut buffer = gst::Buffer::new();
        attach(
            buffer.get_mut().expect("new buffer is writable"),
            "subject",
            None,
            None,
        )
        .expect("metadata attaches");
        let envelope = read(&buffer).expect("metadata reads");
        assert!(envelope.headers.is_none());
        assert!(envelope.reply_subject.is_none());
    }

    #[test]
    fn malformed_message_meta_is_rejected() {
        init();
        let mut buffer = gst::Buffer::new();
        let mut meta = gst::meta::CustomMeta::add(
            buffer.get_mut().expect("new buffer is writable"),
            META_NAME,
        )
        .expect("metadata attaches");
        meta.mut_structure().set("subject", 42_i32);
        assert_eq!(
            read(&buffer).err().as_deref(),
            Some("GstNatsMessageMeta subject is missing or is not a string")
        );
    }

    #[test]
    fn fixed_headers_are_merged_before_message_headers_with_duplicates() {
        let mut fixed = async_nats::HeaderMap::new();
        fixed.append("X-Test", "fixed-one");
        fixed.append("X-Test", "fixed-two");
        let mut message = async_nats::HeaderMap::new();
        message.append("X-Test", "message-one");
        message.append("X-Test", "message-two");

        let merged = merge_headers(Some(&fixed), Some(&message)).expect("merged headers");
        assert_eq!(
            merged
                .get_all("X-Test")
                .map(async_nats::HeaderValue::as_str)
                .collect::<Vec<_>>(),
            ["fixed-one", "fixed-two", "message-one", "message-two"]
        );
        assert!(merge_headers(None, None).is_none());
    }
}
