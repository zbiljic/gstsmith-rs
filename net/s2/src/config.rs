use std::num::NonZeroU32;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use gst::glib;
use gst::prelude::*;
use s2_sdk::error::{
    AppendError, AppendSessionError, ProducerError, ReadError, ReadSessionError, RequestError,
};
use s2_sdk::types::{
    AccountEndpoint, AppendRetryPolicy, BasinEndpoint, BasinName, Compression, FencingToken,
    RetryConfig, S2Config, S2Endpoints, StreamName,
};
use url::{Host, Url};

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
    pub allow_insecure_endpoints: bool,
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
            allow_insecure_endpoints: false,
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
            glib::ParamSpecBoolean::builder("allow-insecure-endpoints")
                .nick("Allow Insecure Endpoints")
                .blurb("Allow credentials and data over remote plaintext HTTP endpoints")
                .default_value(false)
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
            "allow-insecure-endpoints" => set_copy(&mut self.allow_insecure_endpoints, value),
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
            "allow-insecure-endpoints" => Some(self.allow_insecure_endpoints.to_value()),
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
                s2 = s2.with_endpoints(validate_endpoints(
                    account,
                    basin_endpoint,
                    self.allow_insecure_endpoints,
                )?);
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

fn validate_endpoints(
    account: &str,
    basin: &str,
    allow_insecure_endpoints: bool,
) -> Result<S2Endpoints, String> {
    validate_endpoint_policy(account, allow_insecure_endpoints)?;
    validate_endpoint_policy(basin, allow_insecure_endpoints)?;
    let account = AccountEndpoint::new(account)
        .map_err(|_parse_error| "account-endpoint is invalid".to_owned())?;
    let basin =
        BasinEndpoint::new(basin).map_err(|_parse_error| "basin-endpoint is invalid".to_owned())?;
    S2Endpoints::new(account, basin)
        .map_err(|_parse_error| "account-endpoint and basin-endpoint schemes must match".to_owned())
}

fn validate_endpoint_policy(endpoint: &str, allow_insecure_endpoints: bool) -> Result<(), String> {
    let endpoint = parse_endpoint(endpoint)?;
    if !endpoint.username().is_empty() || endpoint.password().is_some() {
        Err("endpoint user-info is not permitted".to_owned())
    } else if endpoint.scheme() == "https"
        || (endpoint.scheme() == "http"
            && (allow_insecure_endpoints || is_loopback_s2_lite_endpoint(&endpoint)))
    {
        Ok(())
    } else {
        Err(
            "endpoint must use HTTPS, or HTTP for a loopback S2 Lite endpoint; set allow-insecure-endpoints=true to permit remote plaintext HTTP"
                .to_owned(),
        )
    }
}

fn parse_endpoint(endpoint: &str) -> Result<Url, String> {
    let endpoint = endpoint.replace("{basin}.", "placeholder.");
    match Url::parse(&endpoint) {
        Ok(endpoint) => Ok(endpoint),
        Err(url::ParseError::RelativeUrlWithoutBase) => Url::parse(&format!("https://{endpoint}"))
            .map_err(|_parse_error| "endpoint is invalid".to_owned()),
        Err(_parse_error) => Err("endpoint is invalid".to_owned()),
    }
}

fn is_loopback_s2_lite_endpoint(endpoint: &Url) -> bool {
    match endpoint.host() {
        Some(Host::Domain(host)) => host == "localhost",
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        None => false,
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

pub(crate) trait SanitizedS2Error {
    fn sanitized_message(&self) -> String;
}

pub(crate) fn sanitized_error<E: SanitizedS2Error>(error: &E) -> String {
    error.sanitized_message()
}

impl SanitizedS2Error for RequestError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Client(_) => "S2 client operation failed".to_owned(),
            Self::MalformedAccessToken(_) => "S2 access token is malformed".to_owned(),
            Self::Validation(_) => "S2 rejected an invalid request".to_owned(),
            Self::Server(response) => format!("S2 server error code {}", response.code),
            _ => "S2 request failed".to_owned(),
        }
    }
}

impl SanitizedS2Error for ReadError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Request(error) => error.sanitized_message(),
            Self::ReadUnwritten(position) => {
                format!(
                    "S2 read started beyond the current tail at sequence {}",
                    position.seq_num
                )
            }
            _ => "S2 read failed".to_owned(),
        }
    }
}

impl SanitizedS2Error for ReadSessionError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Read(error) => error.sanitized_message(),
            Self::HeartbeatTimeout => "S2 read session heartbeat timed out".to_owned(),
            _ => "S2 read session failed".to_owned(),
        }
    }
}

impl SanitizedS2Error for AppendError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Request(error) => error.sanitized_message(),
            Self::ConditionFailed(_) => "S2 append condition failed".to_owned(),
            _ => "S2 append failed".to_owned(),
        }
    }
}

impl SanitizedS2Error for AppendSessionError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Append(error) => error.sanitized_message(),
            Self::AckTimeout => "S2 append acknowledgement timed out".to_owned(),
            Self::ServerDisconnected => "S2 append server disconnected".to_owned(),
            Self::StreamClosedEarly => "S2 append response stream closed early".to_owned(),
            Self::SessionClosed => "S2 append session was already closed".to_owned(),
            Self::SessionClosing => "S2 append session is closing".to_owned(),
            Self::SessionDropped => "S2 append session was dropped".to_owned(),
            Self::InvalidAck(_) => "S2 append acknowledgement was invalid".to_owned(),
            _ => "S2 append session failed".to_owned(),
        }
    }
}

impl SanitizedS2Error for ProducerError {
    fn sanitized_message(&self) -> String {
        match self {
            Self::Append(error) => error.sanitized_message(),
            Self::Validation(_) => "S2 producer input was invalid".to_owned(),
            Self::ProducerClosed => "S2 producer was already closed".to_owned(),
            Self::ProducerClosing => "S2 producer is closing".to_owned(),
            Self::ProducerDropped => "S2 producer was dropped".to_owned(),
            _ => "S2 producer failed".to_owned(),
        }
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
        let malformed = RequestError::MalformedAccessToken(sentinel.to_owned());
        assert!(!sanitized_error(&malformed).contains(sentinel));
    }

    #[test]
    fn endpoint_policy_accepts_https_and_s2_lite_loopback_http() {
        for endpoint in [
            "https://example.test",
            "example.test",
            "https://{basin}.example.test",
            "http://127.0.0.1:8080",
            "http://127.1.2.3:8080",
            "http://[::1]:8080",
            "http://localhost:8080",
        ] {
            validate_endpoint_policy(endpoint, false)
                .unwrap_or_else(|error| panic!("{endpoint} should be accepted: {error}"));
        }
    }

    #[test]
    fn endpoint_policy_requires_opt_in_for_remote_plaintext_http() {
        let error = validate_endpoint_policy("http://example.test", false)
            .expect_err("remote plaintext HTTP must be rejected by default");
        assert!(error.contains("allow-insecure-endpoints=true"));
        validate_endpoint_policy("http://example.test", true)
            .expect("explicit opt-in accepts remote plaintext HTTP");
    }

    #[test]
    fn endpoint_policy_rejects_user_info_without_echoing_it() {
        let user_info = "SENTINEL_ENDPOINT_USER_INFO";
        let error = validate_endpoint_policy(&format!("https://{user_info}@example.test"), false)
            .expect_err("endpoint user-info must be rejected");
        assert_eq!(error, "endpoint user-info is not permitted");
        assert!(!error.contains(user_info));
    }

    #[test]
    fn endpoint_pair_rejects_mixed_schemes() {
        let error = validate_endpoints("https://example.test", "http://127.0.0.1", false)
            .expect_err("the SDK contract rejects mixed endpoint schemes");
        assert_eq!(
            error,
            "account-endpoint and basin-endpoint schemes must match"
        );
    }
}
