use std::time::Duration;

use reqwest::{Client, Url};

use crate::prompt::Message;

mod openai_chat;

pub(crate) struct GenerationRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
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
