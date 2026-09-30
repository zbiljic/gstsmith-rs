use std::time::Duration;

use futures_util::StreamExt;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};

use super::{
    BackendError, GenerationRequest, GenerationResult, ResponseFormat, SamplingMode,
    TokenLimitMode, Usage,
};
use crate::prompt::{Message, Part, Role};

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<WireResponseFormat<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireResponseFormat<'a> {
    Text,
    JsonObject,
    JsonSchema { json_schema: WireSchema<'a> },
}

#[derive(Serialize)]
struct WireSchema<'a> {
    name: &'static str,
    strict: bool,
    schema: &'a serde_json::Value,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: WireContent<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireContent<'a> {
    Text(&'a str),
    Parts(Vec<WirePart<'a>>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum WirePart<'a> {
    #[serde(rename = "text")]
    Text { text: &'a str },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl<'a> },
}

#[derive(Serialize)]
struct ImageUrl<'a> {
    url: &'a str,
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<Choice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    refusal: Option<String>,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Deserialize)]
struct CompletionTokensDetails {
    reasoning_tokens: Option<u64>,
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Model => "assistant",
    }
}

fn wire_message(message: &Message) -> Result<WireMessage<'_>, BackendError> {
    let content = match message.parts.as_slice() {
        [Part::Text(text)] => WireContent::Text(text),
        parts => {
            let wire_parts = parts
                .iter()
                .map(|part| match part {
                    Part::Text(text) => Ok(WirePart::Text { text }),
                    Part::Media {
                        mime_type,
                        data_url,
                    } if mime_type == "image/jpeg" => Ok(WirePart::ImageUrl {
                        image_url: ImageUrl { url: data_url },
                    }),
                    Part::Media { .. } => {
                        Err(BackendError::Response("unsupported prompt media type"))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            WireContent::Parts(wire_parts)
        }
    };
    Ok(WireMessage {
        role: role_name(message.role),
        content,
    })
}

fn serialize(request: &GenerationRequest) -> Result<Vec<u8>, BackendError> {
    let response_format = match request.response_format {
        ResponseFormat::Default => None,
        ResponseFormat::Text => Some(WireResponseFormat::Text),
        ResponseFormat::JsonObject => Some(WireResponseFormat::JsonObject),
        ResponseFormat::JsonSchema => Some(WireResponseFormat::JsonSchema {
            json_schema: WireSchema {
                name: "vlmanalysis",
                strict: true,
                schema: request
                    .response_schema
                    .as_deref()
                    .ok_or(BackendError::Response("response schema is missing"))?,
            },
        }),
    };
    let messages = request
        .messages
        .iter()
        .map(wire_message)
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::to_vec(&WireRequest {
        model: &request.model,
        messages,
        max_tokens: (request.token_limit_mode == TokenLimitMode::Legacy)
            .then_some(request.max_tokens),
        max_completion_tokens: (request.token_limit_mode == TokenLimitMode::Completion)
            .then_some(request.max_tokens),
        temperature: (request.sampling_mode == SamplingMode::Configured)
            .then_some(request.temperature),
        top_p: (request.sampling_mode == SamplingMode::Configured).then_some(request.top_p),
        reasoning_effort: request.reasoning_effort.as_deref(),
        stream: false,
        response_format,
    })
    .map_err(|_error| BackendError::Response("failed to serialize generation request"))
}

fn parse(body: &[u8], format: ResponseFormat) -> Result<GenerationResult, BackendError> {
    let response: WireResponse = serde_json::from_slice(body)
        .map_err(|_error| BackendError::Response("response is not valid Chat Completions JSON"))?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or(BackendError::Response("response has no choices"))?;
    if choice
        .message
        .refusal
        .is_some_and(|refusal| !refusal.is_empty())
    {
        return Err(BackendError::Response("provider refused the request"));
    }
    match choice.finish_reason.as_deref() {
        None | Some("stop") => {}
        Some("length") => {
            return Err(BackendError::Response("response reached the token limit"));
        }
        Some("content_filter") => {
            return Err(BackendError::Response(
                "response was filtered by the provider",
            ));
        }
        Some(_) => {
            return Err(BackendError::Response(
                "response has an unsupported finish reason",
            ));
        }
    }
    let text = choice
        .message
        .content
        .filter(|content| !content.is_empty())
        .ok_or(BackendError::Response(
            "response has no non-empty string message content",
        ))?;
    if matches!(
        format,
        ResponseFormat::JsonObject | ResponseFormat::JsonSchema
    ) {
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|_error| BackendError::Response("response content is not valid JSON"))?;
        if !json.is_object() {
            return Err(BackendError::Response(
                "response content must be a JSON object",
            ));
        }
    }
    let usage = response.usage.map_or(
        Usage {
            prompt: None,
            completion: None,
            reasoning: None,
        },
        |usage| Usage {
            prompt: usage.prompt_tokens,
            completion: usage.completion_tokens,
            reasoning: usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens),
        },
    );
    Ok(GenerationResult { text, usage })
}

enum BodyErrorKind {
    Read,
    TooLarge,
}

struct BodyError {
    bytes_read: usize,
    kind: BodyErrorKind,
}

async fn bounded_body(response: reqwest::Response) -> Result<Vec<u8>, BodyError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(next) = stream.next().await {
        let chunk = next.map_err(|_error| BodyError {
            bytes_read: body.len(),
            kind: BodyErrorKind::Read,
        })?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(BodyError {
                bytes_read: body.len().saturating_add(chunk.len()),
                kind: BodyErrorKind::TooLarge,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

pub(super) async fn generate(
    client: &Client,
    endpoint: Url,
    api_key: Option<&str>,
    request: GenerationRequest,
    timeout: Duration,
) -> Result<GenerationResult, BackendError> {
    let body = serialize(&request)?;
    let operation = async {
        let mut builder = client
            .post(endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);
        if let Some(api_key) = api_key {
            builder = builder.bearer_auth(api_key);
        }
        let response = builder.send().await.map_err(|_error| BackendError::Http {
            status: None,
            body_bytes: None,
            message: "HTTP request failed",
        })?;
        let status = response.status();
        let body = bounded_body(response).await;
        if !status.is_success() {
            let body_bytes = match body {
                Ok(body) => body.len(),
                Err(error) => error.bytes_read,
            };
            return Err(BackendError::Http {
                status: Some(status.as_u16()),
                body_bytes: Some(body_bytes),
                message: status_message(status),
            });
        }
        let body = body.map_err(|error| match error.kind {
            BodyErrorKind::Read => BackendError::Http {
                status: None,
                body_bytes: Some(error.bytes_read),
                message: "failed while reading HTTP response",
            },
            BodyErrorKind::TooLarge => BackendError::Response("response exceeds 1 MiB limit"),
        })?;
        parse(&body, request.response_format)
    };
    tokio::time::timeout(timeout, operation)
        .await
        .map_err(|_elapsed| BackendError::Timeout)?
}

fn status_message(status: StatusCode) -> &'static str {
    if status.is_client_error() {
        "provider returned an HTTP client error"
    } else if status.is_server_error() {
        "provider returned an HTTP server error"
    } else {
        "provider returned an unsuccessful HTTP status"
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use rcgen::{
        BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
        KeyUsagePurpose,
    };
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
    use tokio_rustls::TlsAcceptor;

    use super::*;

    fn generation(messages: Vec<Message>) -> GenerationRequest {
        GenerationRequest {
            model: "model".into(),
            messages,
            max_tokens: 512,
            temperature: 0.2,
            top_p: 0.9,
            response_format: ResponseFormat::Default,
            response_schema: None,
            token_limit_mode: TokenLimitMode::Legacy,
            sampling_mode: SamplingMode::Configured,
            reasoning_effort: None,
        }
    }

    #[test]
    fn openai_chat_maps_all_roles_and_exact_tuning_fields() {
        let messages = [Role::System, Role::User, Role::Model]
            .into_iter()
            .map(|role| Message {
                role,
                parts: vec![Part::Text("literal {{value}}".into())],
            })
            .collect();
        let value: serde_json::Value =
            serde_json::from_slice(&serialize(&generation(messages)).unwrap_or_default())
                .unwrap_or_default();
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["role"], "user");
        assert_eq!(value["messages"][2]["role"], "assistant");
        assert_eq!(value["messages"][0]["content"], "literal {{value}}");
        assert_eq!(value["messages"][1]["content"], "literal {{value}}");
        assert_eq!(value["messages"][2]["content"], "literal {{value}}");
        assert_eq!(value["model"], "model");
        assert_eq!(value["max_tokens"], 512);
        assert_eq!(value["temperature"], 0.2);
        assert_eq!(value["top_p"], 0.9);
        assert_eq!(value["stream"], false);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(6));
        assert!(value.get("response_format").is_none());
    }

    #[test]
    fn openai_chat_serializes_explicit_reasoning_controls() {
        for token_mode in [TokenLimitMode::Legacy, TokenLimitMode::Completion] {
            for sampling_mode in [SamplingMode::Configured, SamplingMode::ProviderDefault] {
                for effort in [None, Some("none"), Some("low"), Some("custom_2-level")] {
                    let mut request = generation(Vec::new());
                    request.token_limit_mode = token_mode;
                    request.sampling_mode = sampling_mode;
                    request.reasoning_effort = effort.map(Arc::from);
                    let mut expected =
                        serde_json::json!({"model":"model", "messages":[], "stream":false});
                    let limit = if token_mode == TokenLimitMode::Legacy {
                        "max_tokens"
                    } else {
                        "max_completion_tokens"
                    };
                    expected[limit] = 512.into();
                    if sampling_mode == SamplingMode::Configured {
                        expected["temperature"] = 0.2.into();
                        expected["top_p"] = 0.9.into();
                    }
                    if let Some(effort) = effort {
                        expected["reasoning_effort"] = effort.into();
                    }
                    let wire: serde_json::Value = serde_json::from_slice(
                        &serialize(&request).expect("serializing reasoning controls"),
                    )
                    .expect("parsing wire request");
                    assert_eq!(wire, expected);
                }
            }
        }
    }

    #[test]
    fn openai_chat_parses_optional_reasoning_usage() {
        for (details, expected) in [
            (None, None),
            (Some(serde_json::Value::Null), None),
            (Some(serde_json::json!({})), None),
            (Some(serde_json::json!({"reasoning_tokens":null})), None),
            (Some(serde_json::json!({"reasoning_tokens":0})), Some(0)),
            (Some(serde_json::json!({"reasoning_tokens":42})), Some(42)),
        ] {
            let mut body = serde_json::json!({"choices":[{"message":{"content":"ok"}}], "usage":{"prompt_tokens":7,"completion_tokens":50}});
            if let Some(details) = details {
                body["usage"]["completion_tokens_details"] = details;
            }
            let result = parse(body.to_string().as_bytes(), ResponseFormat::Default)
                .expect("optional reasoning usage");
            assert_eq!(result.usage.prompt, Some(7));
            assert_eq!(result.usage.completion, Some(50));
            assert_eq!(result.usage.reasoning, expected);
        }
    }

    #[test]
    fn openai_chat_serializes_response_formats() {
        let schema = serde_json::json!({"type": "object", "properties": {}});
        for (format, expected) in [
            (ResponseFormat::Text, serde_json::json!({"type": "text"})),
            (
                ResponseFormat::JsonObject,
                serde_json::json!({"type": "json_object"}),
            ),
            (
                ResponseFormat::JsonSchema,
                serde_json::json!({"type": "json_schema", "json_schema": {"name": "vlmanalysis", "strict": true, "schema": schema}}),
            ),
        ] {
            let mut request = generation(Vec::new());
            request.response_format = format;
            request.response_schema = Some(Arc::new(schema.clone()));
            let wire: serde_json::Value =
                serde_json::from_slice(&serialize(&request).expect("serializing format"))
                    .expect("parsing request");
            assert_eq!(wire["response_format"], expected);
        }
    }

    #[test]
    fn openai_chat_validates_json_objects_after_completion_status() {
        let object = " \n{\"scene\":\"street\"}\t ";
        for format in [
            ResponseFormat::Default,
            ResponseFormat::Text,
            ResponseFormat::JsonObject,
            ResponseFormat::JsonSchema,
        ] {
            for content in [
                object,
                "private malformed JSON",
                "[]",
                "null",
                "12",
                "\"string\"",
            ] {
                let body = serde_json::json!({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}).to_string();
                let result = parse(body.as_bytes(), format);
                if content == object
                    || matches!(format, ResponseFormat::Default | ResponseFormat::Text)
                {
                    assert_eq!(result.expect("accepted response").text, content);
                } else {
                    assert!(matches!(result, Err(BackendError::Response(_))));
                }
            }
            let partial = br#"{"choices":[{"message":{"content":"{"},"finish_reason":"length"}]}"#;
            assert!(matches!(
                parse(partial, format),
                Err(BackendError::Response("response reached the token limit"))
            ));
        }
    }

    #[test]
    fn openai_chat_serializes_one_and_two_frames_in_order() {
        for count in [1, 2] {
            let urls = (0..count).map(|n| format!("data:{n}")).collect();
            let messages = crate::prompt::literal_messages(None, "user".into(), urls);
            let value: serde_json::Value =
                serde_json::from_slice(&serialize(&generation(messages)).unwrap_or_default())
                    .unwrap_or_default();
            assert_eq!(
                value["messages"][0]["content"].as_array().map(Vec::len),
                Some(count + 1)
            );
            assert_eq!(value["messages"][0]["content"][0]["type"], "text");
            assert_eq!(value["messages"][0]["content"][0]["text"], "user");
            for index in 0..count {
                assert_eq!(
                    value["messages"][0]["content"][index + 1]["type"],
                    "image_url"
                );
                assert_eq!(
                    value["messages"][0]["content"][index + 1]["image_url"]["url"],
                    format!("data:{index}")
                );
            }
        }
    }

    #[test]
    fn openai_chat_parses_usage_presence_and_absence() {
        let with = parse(br#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#, ResponseFormat::Default).expect("parsing response with usage");
        assert_eq!(with.usage.prompt, Some(2));
        let without = parse(
            br#"{"choices":[{"message":{"content":"ok"}}]}"#,
            ResponseFormat::Default,
        )
        .expect("parsing response without usage");
        assert_eq!(without.usage.completion, None);
    }

    #[test]
    fn openai_chat_rejects_malformed_missing_non_string_and_empty_content() {
        for body in [
            br"not json".as_slice(),
            br#"{"choices":[]}"#,
            br#"{"choices":[{"message":{"content":12}}]}"#,
            br#"{"choices":[{"message":{}}]}"#,
            br#"{"choices":[{"message":{"content":null}}]}"#,
            br#"{"choices":[{"message":{"content":""}}]}"#,
            br#"{"choices":[{"message":{"content":""}},{"message":{"content":"later"}}]}"#,
        ] {
            assert!(parse(body, ResponseFormat::Default).is_err());
        }
    }

    #[test]
    fn openai_chat_accepts_stop_and_omitted_or_null_finish_reason() {
        for fields in [
            "",
            r#", "finish_reason":null"#,
            r#", "finish_reason":"stop""#,
        ] {
            let body =
                format!(r#"{{"choices":[{{"message":{{"content":"ok","refusal":""}}{fields}}}]}}"#);
            assert_eq!(
                parse(body.as_bytes(), ResponseFormat::Default)
                    .expect("complete response")
                    .text,
                "ok"
            );
        }
    }

    #[test]
    fn openai_chat_rejects_refusal_before_content_and_finish_reason() {
        for content in ["null", r#""partial""#] {
            let body = format!(
                r#"{{"choices":[{{"message":{{"content":{content},"refusal":"private refusal"}},"finish_reason":"length"}}]}}"#
            );
            assert!(matches!(
                parse(body.as_bytes(), ResponseFormat::Default),
                Err(BackendError::Response("provider refused the request"))
            ));
        }
    }

    #[test]
    fn openai_chat_rejects_incomplete_and_unsupported_finish_reasons() {
        for (reason, expected) in [
            ("length", "response reached the token limit"),
            ("content_filter", "response was filtered by the provider"),
            ("tool_calls", "response has an unsupported finish reason"),
            ("function_call", "response has an unsupported finish reason"),
            (
                "private unknown reason",
                "response has an unsupported finish reason",
            ),
        ] {
            let body = serde_json::json!({"choices": [{"message": {"content": "{\"partial\":"}, "finish_reason": reason}]}).to_string();
            assert!(
                matches!(parse(body.as_bytes(), ResponseFormat::Default), Err(BackendError::Response(message)) if message == expected)
            );
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the local trusted and untrusted TLS paths intentionally share one certificate/server fixture"
    )]
    fn https_trusted_local_ca_succeeds_and_untrusted_chain_fails() {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            let _install_result = rustls::crypto::ring::default_provider().install_default();
        }
        let mut ca_params =
            CertificateParams::new(Vec::<String>::new()).expect("constructing test CA parameters");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
        ];
        let ca_key = KeyPair::generate().expect("generating test CA key");
        let ca_certificate = ca_params
            .self_signed(&ca_key)
            .expect("self-signing test CA");
        let issuer = Issuer::new(ca_params, ca_key);

        let mut server_params = CertificateParams::new(vec!["127.0.0.1".to_owned()])
            .expect("constructing test server parameters");
        server_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        server_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let server_key = KeyPair::generate().expect("generating test server key");
        let server_certificate = server_params
            .signed_by(&server_key, &issuer)
            .expect("signing test server certificate");
        let certificate_chain = vec![
            CertificateDer::from(server_certificate.der().to_vec()),
            CertificateDer::from(ca_certificate.der().to_vec()),
        ];
        let private_key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(server_key.serialize_der()));
        let server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certificate_chain, private_key)
            .expect("building test TLS server configuration");

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("building HTTPS test runtime");
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("binding HTTPS test server");
            let address = listener.local_addr().expect("reading HTTPS test address");
            let acceptor = TlsAcceptor::from(Arc::new(server_config));
            let server = tokio::spawn(async move {
                for _connection in 0..2 {
                    let (stream, _peer) = listener.accept().await.expect("accepting HTTPS client");
                    let acceptor = acceptor.clone();
                    tokio::spawn(async move {
                        let Ok(stream) = acceptor.accept(stream).await else {
                            return;
                        };
                        let service =
                            service_fn(|_request: Request<hyper::body::Incoming>| async {
                                Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
                                    br#"{"choices":[{"message":{"content":"tls-ok"}}]}"#,
                                ))))
                            });
                        let _connection_result = hyper::server::conn::http1::Builder::new()
                            .serve_connection(TokioIo::new(stream), service)
                            .await;
                    });
                }
            });
            let endpoint = Url::parse(&format!("https://{address}/v1/chat/completions"))
                .expect("building HTTPS endpoint URL");
            let root = reqwest::Certificate::from_der(ca_certificate.der().as_ref())
                .expect("parsing test root certificate");
            let trusted_client = Client::builder()
                .add_root_certificate(root)
                .build()
                .expect("building trusted HTTPS client");
            let trusted = generate(
                &trusted_client,
                endpoint.clone(),
                None,
                generation(crate::prompt::literal_messages(
                    None,
                    "describe".into(),
                    vec!["data:image/jpeg;base64,/9j/2Q==".into()],
                )),
                Duration::from_secs(2),
            )
            .await
            .expect("trusted test chain succeeds");
            assert_eq!(trusted.text, "tls-ok");

            let untrusted_client = Client::builder()
                .build()
                .expect("building normal Web PKI client");
            let untrusted = generate(
                &untrusted_client,
                endpoint,
                None,
                generation(crate::prompt::literal_messages(
                    None,
                    "describe".into(),
                    vec!["data:image/jpeg;base64,/9j/2Q==".into()],
                )),
                Duration::from_secs(2),
            )
            .await;
            assert!(matches!(
                untrusted,
                Err(BackendError::Http {
                    status: None,
                    body_bytes: None,
                    ..
                })
            ));
            server.await.expect("joining HTTPS accept loop");
        });
    }
}
