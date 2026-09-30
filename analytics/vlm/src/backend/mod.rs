use std::sync::Arc;
use std::time::Duration;

use gst::glib;
use reqwest::{Client, Url};

use crate::prompt::Message;

mod openai_chat;

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[repr(i32)]
#[enum_type(name = "GstSmithVlmResponseFormat")]
pub(crate) enum ResponseFormat {
    #[default]
    #[enum_value(name = "Provider default", nick = "default")]
    Default = 0,
    #[enum_value(name = "Text", nick = "text")]
    Text = 1,
    #[enum_value(name = "JSON object", nick = "json-object")]
    JsonObject = 2,
    #[enum_value(name = "JSON schema", nick = "json-schema")]
    JsonSchema = 3,
}

pub(crate) struct GenerationRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) response_format: ResponseFormat,
    pub(crate) response_schema: Option<Arc<serde_json::Value>>,
}

pub(crate) struct Usage {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
}

pub(crate) struct GenerationResult {
    pub(crate) text: String,
    pub(crate) usage: Usage,
}

#[derive(Debug)]
pub(crate) enum BackendError {
    Timeout,
    Http {
        status: Option<u16>,
        body_bytes: Option<usize>,
        message: &'static str,
    },
    Response(&'static str),
}

pub(crate) async fn generate(
    client: &Client,
    endpoint: Url,
    api_key: Option<&str>,
    request: GenerationRequest,
    timeout: Duration,
) -> Result<GenerationResult, BackendError> {
    openai_chat::generate(client, endpoint, api_key, request, timeout).await
}
