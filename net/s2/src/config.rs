use std::num::NonZeroU32;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use gst::glib;
use gst::prelude::*;
use s2_sdk::types::{
    AccountEndpoint, AppendRetryPolicy, BasinEndpoint, BasinName, Compression, FencingToken,
    RetryConfig, S2Config, S2Endpoints, StreamName,
};

pub const DEFAULT_CONNECTION_TIMEOUT: u64 = 3_000_000_000;
pub const DEFAULT_REQUEST_TIMEOUT: u64 = 5_000_000_000;
pub const DEFAULT_RETRY_MAX_ATTEMPTS: u32 = 3;
pub const DEFAULT_RETRY_MIN_DELAY: u64 = 100_000_000;
pub const DEFAULT_RETRY_MAX_DELAY: u64 = 1_000_000_000;
pub const DEFAULT_QUEUE_CAPACITY: u32 = 64;

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[enum_type(name = "GstS2Compression")]
pub enum CompressionMode {
    #[default]
    #[enum_value(name = "None", nick = "none")]
    None,
    #[enum_value(name = "Gzip", nick = "gzip")]
    Gzip,
    #[enum_value(name = "Zstandard", nick = "zstd")]
    Zstd,
}

impl From<CompressionMode> for Compression {
    fn from(value: CompressionMode) -> Self {
        match value {
            CompressionMode::None => Self::None,
            CompressionMode::Gzip => Self::Gzip,
            CompressionMode::Zstd => Self::Zstd,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[enum_type(name = "GstS2AppendRetryPolicy")]
pub enum SinkAppendRetryPolicy {
    #[default]
    #[enum_value(name = "No side effects", nick = "no-side-effects")]
    NoSideEffects,
    #[enum_value(name = "All", nick = "all")]
    All,
}

impl From<SinkAppendRetryPolicy> for AppendRetryPolicy {
    fn from(value: SinkAppendRetryPolicy) -> Self {
        match value {
            SinkAppendRetryPolicy::NoSideEffects => Self::NoSideEffects,
            SinkAppendRetryPolicy::All => Self::All,
        }
    }
}

#[derive(Clone)]
pub struct ConnectionSettings {
    pub basin: Option<String>,
    pub stream: Option<String>,
    pub access_token_file: Option<String>,
    pub account_endpoint: Option<String>,
    pub basin_endpoint: Option<String>,
    pub connection_timeout: u64,
    pub request_timeout: u64,
    pub retry_max_attempts: u32,
    pub retry_min_delay: u64,
    pub retry_max_delay: u64,
    pub compression: CompressionMode,
    pub queue_capacity: u32,
}

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self {
            basin: None,
            stream: None,
            access_token_file: None,
            account_endpoint: None,
            basin_endpoint: None,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            retry_max_attempts: DEFAULT_RETRY_MAX_ATTEMPTS,
            retry_min_delay: DEFAULT_RETRY_MIN_DELAY,
            retry_max_delay: DEFAULT_RETRY_MAX_DELAY,
            compression: CompressionMode::None,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
        }
    }
}

#[derive(Clone)]
pub struct ValidatedConnection {
    pub basin: BasinName,
    pub stream: StreamName,
    pub s2: S2Config,
    pub queue_capacity: usize,
}

impl ConnectionSettings {
    pub fn property_specs() -> Vec<glib::ParamSpec> {
        vec![
            glib::ParamSpecString::builder("basin")
                .nick("Basin")
                .blurb("S2 basin name")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("stream")
                .nick("Stream")
                .blurb("S2 stream name")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("access-token-file")
                .nick("Access Token File")
                .blurb("File containing the S2 access token; otherwise S2_ACCESS_TOKEN is used")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("account-endpoint")
                .nick("Account Endpoint")
                .blurb("Optional explicit S2 account endpoint")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("basin-endpoint")
                .nick("Basin Endpoint")
                .blurb("Optional explicit S2 basin endpoint")
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt64::builder("connection-timeout")
                .nick("Connection Timeout")
                .blurb("S2 connection timeout in nanoseconds")
                .minimum(1)
                .default_value(DEFAULT_CONNECTION_TIMEOUT)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt64::builder("request-timeout")
                .nick("Request Timeout")
                .blurb("S2 request timeout in nanoseconds")
                .minimum(1)
                .default_value(DEFAULT_REQUEST_TIMEOUT)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt::builder("retry-max-attempts")
                .nick("Retry Maximum Attempts")
                .blurb("Total S2 request attempts including the initial attempt")
                .minimum(1)
                .default_value(DEFAULT_RETRY_MAX_ATTEMPTS)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt64::builder("retry-min-delay")
                .nick("Retry Minimum Delay")
                .blurb("Minimum S2 retry base delay in nanoseconds")
                .default_value(DEFAULT_RETRY_MIN_DELAY)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt64::builder("retry-max-delay")
                .nick("Retry Maximum Delay")
                .blurb("Maximum S2 retry base delay in nanoseconds")
                .default_value(DEFAULT_RETRY_MAX_DELAY)
                .mutable_ready()
                .build(),
            glib::ParamSpecEnum::builder::<CompressionMode>("compression")
                .nick("Compression")
                .blurb("S2 request and response compression")
                .default_value(CompressionMode::None)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt::builder("queue-capacity")
                .nick("Queue Capacity")
                .blurb("Maximum records waiting between GStreamer and the S2 worker")
                .minimum(1)
                .default_value(DEFAULT_QUEUE_CAPACITY)
                .mutable_ready()
                .build(),
        ]
    }

    pub fn set_property(&mut self, value: &glib::Value, pspec: &glib::ParamSpec) -> bool {
        match pspec.name() {
            "basin" => set_optional_string(&mut self.basin, value),
            "stream" => set_optional_string(&mut self.stream, value),
            "access-token-file" => set_optional_string(&mut self.access_token_file, value),
            "account-endpoint" => set_optional_string(&mut self.account_endpoint, value),
            "basin-endpoint" => set_optional_string(&mut self.basin_endpoint, value),
            "connection-timeout" => set_copy(&mut self.connection_timeout, value),
            "request-timeout" => set_copy(&mut self.request_timeout, value),
            "retry-max-attempts" => set_copy(&mut self.retry_max_attempts, value),
            "retry-min-delay" => set_copy(&mut self.retry_min_delay, value),
            "retry-max-delay" => set_copy(&mut self.retry_max_delay, value),
            "compression" => set_copy(&mut self.compression, value),
            "queue-capacity" => set_copy(&mut self.queue_capacity, value),
            _ => false,
        }
    }

    pub fn property(&self, pspec: &glib::ParamSpec) -> Option<glib::Value> {
        match pspec.name() {
            "basin" => Some(self.basin.to_value()),
            "stream" => Some(self.stream.to_value()),
            "access-token-file" => Some(self.access_token_file.to_value()),
            "account-endpoint" => Some(self.account_endpoint.to_value()),
            "basin-endpoint" => Some(self.basin_endpoint.to_value()),
            "connection-timeout" => Some(self.connection_timeout.to_value()),
            "request-timeout" => Some(self.request_timeout.to_value()),
            "retry-max-attempts" => Some(self.retry_max_attempts.to_value()),
            "retry-min-delay" => Some(self.retry_min_delay.to_value()),
            "retry-max-delay" => Some(self.retry_max_delay.to_value()),
            "compression" => Some(self.compression.to_value()),
            "queue-capacity" => Some(self.queue_capacity.to_value()),
            _ => None,
        }
    }

    pub fn validate(
        &self,
        append_policy: AppendRetryPolicy,
    ) -> Result<ValidatedConnection, String> {
        let basin = self
            .basin
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "basin must be set".to_owned())?
            .parse::<BasinName>()
            .map_err(|_parse_error| "basin is not a valid S2 basin name".to_owned())?;
        let stream = self
            .stream
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "stream must be set".to_owned())?
            .parse::<StreamName>()
            .map_err(|_parse_error| "stream is not a valid S2 stream name".to_owned())?;
        if self.retry_max_delay < self.retry_min_delay {
            return Err("retry-max-delay must be at least retry-min-delay".to_owned());
        }
        let attempts = NonZeroU32::new(self.retry_max_attempts)
            .ok_or_else(|| "retry-max-attempts must be at least one".to_owned())?;
        let queue_capacity = usize::try_from(self.queue_capacity).map_err(|_conversion_error| {
            "queue-capacity is not representable on this platform".to_owned()
        })?;
        let token = load_access_token(self.access_token_file.as_deref())?;
        let retry = RetryConfig::new()
            .with_max_attempts(attempts)
            .with_min_base_delay(Duration::from_nanos(self.retry_min_delay))
            .with_max_base_delay(Duration::from_nanos(self.retry_max_delay))
            .with_append_retry_policy(append_policy);
        let mut s2 = S2Config::new(token)
            .with_rustls_ring_crypto_provider()
            .with_connection_timeout(Duration::from_nanos(self.connection_timeout))
            .with_request_timeout(Duration::from_nanos(self.request_timeout))
            .with_retry(retry)
            .with_compression(self.compression.into());
        match (
            self.account_endpoint
                .as_deref()
                .filter(|value| !value.is_empty()),
            self.basin_endpoint
                .as_deref()
                .filter(|value| !value.is_empty()),
        ) {
            (None, None) => {}
            (Some(account), Some(basin_endpoint)) => {
                reject_endpoint_user_info(account)?;
                reject_endpoint_user_info(basin_endpoint)?;
                let account = AccountEndpoint::new(account)
                    .map_err(|_parse_error| "account-endpoint is invalid".to_owned())?;
                let basin_endpoint = BasinEndpoint::new(basin_endpoint)
                    .map_err(|_parse_error| "basin-endpoint is invalid".to_owned())?;
                let endpoints =
                    S2Endpoints::new(account, basin_endpoint).map_err(|_parse_error| {
                        "account-endpoint and basin-endpoint schemes must match".to_owned()
                    })?;
                s2 = s2.with_endpoints(endpoints);
            }
            _ => {
                return Err("account-endpoint and basin-endpoint must be set together".to_owned());
            }
        }
        Ok(ValidatedConnection {
            basin,
            stream,
            s2,
            queue_capacity,
        })
    }
}

fn set_optional_string(slot: &mut Option<String>, value: &glib::Value) -> bool {
    if let Ok(new_value) = value.get::<Option<String>>() {
        *slot = new_value.filter(|text| !text.is_empty());
    }
    true
}

fn set_copy<T: Copy + for<'a> glib::value::FromValue<'a>>(
    slot: &mut T,
    value: &glib::Value,
) -> bool {
    if let Ok(new_value) = value.get::<T>() {
        *slot = new_value;
    }
    true
}

fn reject_endpoint_user_info(endpoint: &str) -> Result<(), String> {
    let authority = endpoint
        .split_once("://")
        .map_or(endpoint, |(_, authority)| authority);
    if authority.contains('@') {
        Err("endpoint user-info is not permitted".to_owned())
    } else {
        Ok(())
    }
}

fn load_access_token(path: Option<&str>) -> Result<String, String> {
    match path {
        Some(path) => load_secret_file(Path::new(path), "access-token-file"),
        None => std::env::var("S2_ACCESS_TOKEN").map_err(|error| match error {
            std::env::VarError::NotPresent => {
                "S2_ACCESS_TOKEN is not set and access-token-file is unset".to_owned()
            }
            std::env::VarError::NotUnicode(_) => "S2_ACCESS_TOKEN is not valid UTF-8".to_owned(),
        }),
    }
    .and_then(|token| {
        if token.is_empty() {
            Err("S2 access token must not be empty".to_owned())
        } else {
            Ok(token)
        }
    })
}

pub fn load_fencing_token(path: Option<&str>) -> Result<Option<FencingToken>, String> {
    path.map(|path| {
        load_secret_file(Path::new(path), "fencing-token-file").and_then(|token| {
            FencingToken::from_str(&token).map_err(|_parse_error| {
                "fencing-token-file contains an invalid fencing token".to_owned()
            })
        })
    })
    .transpose()
}

fn load_secret_file(path: &Path, property: &str) -> Result<String, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("failed to read {property}: {error}"))?;
    let mut text = String::from_utf8(bytes)
        .map_err(|_utf8_error| format!("{property} must contain valid UTF-8"))?;
    if text.ends_with("\r\n") {
        let new_len = text.len().saturating_sub(2);
        text.truncate(new_len);
    } else if text.ends_with('\n') {
        let new_len = text.len().saturating_sub(1);
        text.truncate(new_len);
    }
    if text.is_empty() {
        Err(format!("{property} must not be empty"))
    } else {
        Ok(text)
    }
}

pub fn sanitized_error(error: &s2_sdk::types::S2Error) -> String {
    use s2_sdk::types::S2Error;
    match error {
        S2Error::Client(_) => "S2 client operation failed".to_owned(),
        S2Error::MalformedAccessToken(_) => "S2 access token is malformed".to_owned(),
        S2Error::Validation(_) => "S2 rejected an invalid request".to_owned(),
        S2Error::AppendConditionFailed(_) => "S2 append condition failed".to_owned(),
        S2Error::ReadUnwritten(position) => {
            format!(
                "S2 read started beyond the current tail at sequence {}",
                position.seq_num
            )
        }
        S2Error::Server(response) => format!("S2 server error code {}", response.code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secret_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gstsmith-s2-config-{label}-{}", std::process::id()))
    }

    #[test]
    fn secret_file_removes_only_one_line_ending() {
        for (label, contents, expected) in [
            ("lf", b" token \n".as_slice(), " token "),
            ("crlf", b"token\r\n".as_slice(), "token"),
            ("double", b"token\n\n".as_slice(), "token\n"),
        ] {
            let path = secret_path(label);
            std::fs::write(&path, contents).expect("writing secret test file");
            assert_eq!(
                load_secret_file(&path, "test-secret").expect("reading test secret"),
                expected
            );
            std::fs::remove_file(path).expect("removing secret test file");
        }
    }

    #[test]
    fn secret_file_rejects_empty_and_non_utf8_content_without_echoing_it() {
        for (label, contents) in [
            ("empty", Vec::new()),
            ("newline", b"\n".to_vec()),
            ("binary", vec![0xff, 0xfe]),
        ] {
            let path = secret_path(label);
            std::fs::write(&path, contents).expect("writing invalid secret test file");
            let error = load_secret_file(&path, "sentinel-secret-file")
                .expect_err("invalid secret is rejected");
            assert!(!error.contains('\u{fffd}'));
            std::fs::remove_file(path).expect("removing invalid secret test file");
        }
    }

    #[test]
    fn sanitized_sdk_errors_never_include_sensitive_external_text() {
        let sentinel = "SENTINEL_RAW_SECRET";
        let malformed = s2_sdk::types::S2Error::MalformedAccessToken(sentinel.to_owned());
        assert!(!sanitized_error(&malformed).contains(sentinel));
    }

    #[test]
    fn endpoint_user_info_is_rejected() {
        assert!(reject_endpoint_user_info("https://user@example.test").is_err());
        reject_endpoint_user_info("http://example.test").expect("plain endpoint is accepted");
    }
}
