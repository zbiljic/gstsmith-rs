use std::path::PathBuf;
use std::time::Duration;

use gst::glib;
use gst::prelude::*;

pub const DEFAULT_SERVERS: &str = "nats://127.0.0.1:4222";
pub const DEFAULT_CONNECTION_TIMEOUT: u64 = 5_000_000_000;

#[derive(Clone, Debug)]
pub struct ConnectionSettings {
    pub servers: String,
    pub connection_name: Option<String>,
    pub credentials_file: Option<String>,
    pub nkey_file: Option<String>,
    pub tls_required: bool,
    pub tls_ca_file: Option<String>,
    pub tls_client_cert_file: Option<String>,
    pub tls_client_key_file: Option<String>,
    pub connection_timeout: u64,
    pub max_reconnects: u32,
    pub retry_on_initial_connect: bool,
}

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self {
            servers: DEFAULT_SERVERS.to_owned(),
            connection_name: None,
            credentials_file: None,
            nkey_file: None,
            tls_required: false,
            tls_ca_file: None,
            tls_client_cert_file: None,
            tls_client_key_file: None,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
            max_reconnects: 0,
            retry_on_initial_connect: false,
        }
    }
}

impl ConnectionSettings {
    pub fn property_specs() -> Vec<glib::ParamSpec> {
        vec![
            glib::ParamSpecString::builder("servers")
                .nick("Servers")
                .blurb("Comma-separated Core NATS server URLs without user-info")
                .default_value(Some(DEFAULT_SERVERS))
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("connection-name")
                .nick("Connection Name")
                .blurb("Optional NATS monitoring connection name")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("credentials-file")
                .nick("Credentials File")
                .blurb("NATS user-credentials file")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("nkey-file")
                .nick("NKey File")
                .blurb("File containing an NKey seed")
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("tls-required")
                .nick("TLS Required")
                .blurb("Require TLS for the NATS connection")
                .default_value(false)
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("tls-ca-file")
                .nick("TLS CA File")
                .blurb("PEM root CA bundle")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("tls-client-cert-file")
                .nick("TLS Client Certificate File")
                .blurb("PEM client certificate chain")
                .mutable_ready()
                .build(),
            glib::ParamSpecString::builder("tls-client-key-file")
                .nick("TLS Client Key File")
                .blurb("PEM client private key")
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt64::builder("connection-timeout")
                .nick("Connection Timeout")
                .blurb("Initial connection timeout in nanoseconds")
                .minimum(1)
                .default_value(DEFAULT_CONNECTION_TIMEOUT)
                .mutable_ready()
                .build(),
            glib::ParamSpecUInt::builder("max-reconnects")
                .nick("Maximum Reconnects")
                .blurb("Maximum consecutive reconnect attempts; zero means unlimited")
                .default_value(0)
                .mutable_ready()
                .build(),
            glib::ParamSpecBoolean::builder("retry-on-initial-connect")
                .nick("Retry Initial Connection")
                .blurb("Establish the initial connection in the background")
                .default_value(false)
                .mutable_ready()
                .build(),
        ]
    }

    pub fn set_property(&mut self, value: &glib::Value, pspec: &glib::ParamSpec) -> bool {
        match pspec.name() {
            "servers" => value
                .get::<String>()
                .map(|servers| self.servers = servers)
                .is_ok(),
            "connection-name" => value
                .get::<Option<String>>()
                .map(|name| self.connection_name = normalize_optional(name))
                .is_ok(),
            "credentials-file" => value
                .get::<Option<String>>()
                .map(|path| self.credentials_file = normalize_optional(path))
                .is_ok(),
            "nkey-file" => value
                .get::<Option<String>>()
                .map(|path| self.nkey_file = normalize_optional(path))
                .is_ok(),
            "tls-required" => value
                .get::<bool>()
                .map(|required| self.tls_required = required)
                .is_ok(),
            "tls-ca-file" => value
                .get::<Option<String>>()
                .map(|path| self.tls_ca_file = normalize_optional(path))
                .is_ok(),
            "tls-client-cert-file" => value
                .get::<Option<String>>()
                .map(|path| self.tls_client_cert_file = normalize_optional(path))
                .is_ok(),
            "tls-client-key-file" => value
                .get::<Option<String>>()
                .map(|path| self.tls_client_key_file = normalize_optional(path))
                .is_ok(),
            "connection-timeout" => value
                .get::<u64>()
                .map(|timeout| self.connection_timeout = timeout)
                .is_ok(),
            "max-reconnects" => value
                .get::<u32>()
                .map(|attempts| self.max_reconnects = attempts)
                .is_ok(),
            "retry-on-initial-connect" => value
                .get::<bool>()
                .map(|retry| self.retry_on_initial_connect = retry)
                .is_ok(),
            _ => false,
        }
    }

    pub fn property(&self, pspec: &glib::ParamSpec) -> Option<glib::Value> {
        match pspec.name() {
            "servers" => Some(self.servers.to_value()),
            "connection-name" => Some(self.connection_name.to_value()),
            "credentials-file" => Some(self.credentials_file.to_value()),
            "nkey-file" => Some(self.nkey_file.to_value()),
            "tls-required" => Some(self.tls_required.to_value()),
            "tls-ca-file" => Some(self.tls_ca_file.to_value()),
            "tls-client-cert-file" => Some(self.tls_client_cert_file.to_value()),
            "tls-client-key-file" => Some(self.tls_client_key_file.to_value()),
            "connection-timeout" => Some(self.connection_timeout.to_value()),
            "max-reconnects" => Some(self.max_reconnects.to_value()),
            "retry-on-initial-connect" => Some(self.retry_on_initial_connect.to_value()),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<ValidatedConnection, String> {
        if self.credentials_file.is_some() && self.nkey_file.is_some() {
            return Err("credentials-file and nkey-file are mutually exclusive".to_owned());
        }
        if self.tls_client_cert_file.is_some() != self.tls_client_key_file.is_some() {
            return Err(
                "tls-client-cert-file and tls-client-key-file must be configured together"
                    .to_owned(),
            );
        }
        if self.connection_timeout == 0 {
            return Err("connection-timeout must be at least one nanosecond".to_owned());
        }

        let servers = self
            .servers
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                let server = entry
                    .parse::<async_nats::ServerAddr>()
                    .map_err(|_parse_error| {
                        "servers contains a malformed NATS server URL".to_owned()
                    })?;
                if server.username().is_some() || server.password().is_some() {
                    return Err("servers must not contain URL user-info".to_owned());
                }
                Ok(server)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if servers.is_empty() {
            return Err("servers must contain at least one NATS server URL".to_owned());
        }

        validate_readable(self.credentials_file.as_deref(), "credentials-file")?;
        validate_readable(self.nkey_file.as_deref(), "nkey-file")?;
        validate_readable(self.tls_ca_file.as_deref(), "tls-ca-file")?;
        validate_readable(self.tls_client_cert_file.as_deref(), "tls-client-cert-file")?;
        validate_readable(self.tls_client_key_file.as_deref(), "tls-client-key-file")?;

        Ok(ValidatedConnection {
            servers,
            timeout: Duration::from_nanos(self.connection_timeout),
        })
    }

    pub fn options(
        &self,
        validated: &ValidatedConnection,
        fallback_name: &str,
        subscription_capacity: usize,
    ) -> Result<async_nats::ConnectOptions, String> {
        let name = self
            .connection_name
            .as_deref()
            .filter(|name| !name.is_empty())
            .unwrap_or(fallback_name);
        let reconnects = usize::try_from(self.max_reconnects).map_err(|_conversion_error| {
            "max-reconnects is not representable on this platform".to_owned()
        })?;

        let mut options = async_nats::ConnectOptions::new()
            .name(name)
            .connection_timeout(validated.timeout)
            .max_reconnects(if reconnects == 0 {
                None
            } else {
                Some(reconnects)
            })
            .require_tls(self.tls_required)
            .subscription_capacity(subscription_capacity);

        if self.retry_on_initial_connect {
            options = options.retry_on_initial_connect();
        }
        if let Some(path) = self.tls_ca_file.as_ref() {
            options = options.add_root_certificates(PathBuf::from(path));
        }
        if let (Some(cert), Some(key)) = (
            self.tls_client_cert_file.as_ref(),
            self.tls_client_key_file.as_ref(),
        ) {
            options = options.add_client_certificate(PathBuf::from(cert), PathBuf::from(key));
        }
        if let Some(path) = self.credentials_file.as_ref() {
            let contents = std::fs::read_to_string(path)
                .map_err(|error| format!("failed to read credentials-file: {error}"))?;
            options = options
                .credentials(&contents)
                .map_err(|error| format!("failed to parse credentials-file: {error}"))?;
        } else if let Some(path) = self.nkey_file.as_ref() {
            let seed = std::fs::read_to_string(path)
                .map_err(|error| format!("failed to read nkey-file: {error}"))?;
            options = options.nkey(seed.trim().to_owned());
        }

        Ok(options)
    }
}

pub fn observe_events<T>(
    options: async_nats::ConnectOptions,
    weak: glib::WeakRef<T>,
) -> async_nats::ConnectOptions
where
    T: glib::object::ObjectType + IsA<gst::Element> + Send + Sync + 'static,
{
    options.event_callback(move |event| {
        let element = weak.upgrade();
        async move {
            let Some(element) = element else {
                return;
            };
            match event {
                async_nats::Event::Connected => {
                    gst::info!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS connection established"
                    );
                }
                async_nats::Event::Disconnected => {
                    gst::warning!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS connection interrupted"
                    );
                }
                async_nats::Event::LameDuckMode => {
                    gst::warning!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS server entered lame-duck mode"
                    );
                }
                async_nats::Event::Draining => {
                    gst::debug!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS connection is draining"
                    );
                }
                async_nats::Event::Closed => {
                    gst::info!(gst::CAT_RUST, obj = element, "Core NATS connection closed");
                }
                async_nats::Event::SlowConsumer(_) => {
                    gst::warning!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS reported a slow subscription consumer"
                    );
                }
                async_nats::Event::ServerError(_) => {
                    gst::warning!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS server reported an error"
                    );
                }
                async_nats::Event::ClientError(_) => {
                    gst::warning!(
                        gst::CAT_RUST,
                        obj = element,
                        "Core NATS client reported an error"
                    );
                }
            }
        }
    })
}

pub struct ValidatedConnection {
    pub servers: Vec<async_nats::ServerAddr>,
    pub timeout: Duration,
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value.filter(|text| !text.is_empty())
}

fn validate_readable(path: Option<&str>, property: &str) -> Result<(), String> {
    if let Some(path) = path {
        std::fs::File::open(path)
            .map(|_| ())
            .map_err(|error| format!("failed to open {property}: {error}"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        let settings = ConnectionSettings::default();
        let validated = settings.validate().expect("default settings validate");
        assert_eq!(validated.servers.len(), 1);
        assert_eq!(validated.timeout, Duration::from_secs(5));
    }

    #[test]
    fn parses_multiple_servers() {
        let settings = ConnectionSettings {
            servers: "nats://127.0.0.1:4222, nats://localhost:4223".to_owned(),
            ..ConnectionSettings::default()
        };
        assert_eq!(
            settings.validate().expect("servers validate").servers.len(),
            2
        );
    }

    #[test]
    fn accepts_supported_server_urls_and_clusters() {
        for servers in [
            "nats://localhost:4222",
            "tls://localhost:4222",
            "ws://localhost:8080",
            "wss://localhost:8443",
            "nats://localhost:4222, tls://localhost:4223, ws://localhost:8080, wss://localhost:8443",
        ] {
            let settings = ConnectionSettings {
                servers: servers.to_owned(),
                ..ConnectionSettings::default()
            };
            settings
                .validate()
                .expect("supported server URL should validate");
        }
    }

    #[test]
    fn rejects_server_url_user_info_without_echoing_it() {
        const USER_PLACEHOLDER: &str = "user-placeholder";
        const PASSWORD_PLACEHOLDER: &str = "password-placeholder";

        for servers in [
            "nats://user-placeholder@localhost:4222",
            "nats://user-placeholder:password-placeholder@localhost:4222",
            "tls://user-placeholder:password-placeholder@localhost:4222",
            "ws://user-placeholder:password-placeholder@localhost:8080",
            "wss://user-placeholder:password-placeholder@localhost:8443",
            "nats://localhost:4222, ws://user-placeholder:password-placeholder@localhost:8080",
        ] {
            let settings = ConnectionSettings {
                servers: servers.to_owned(),
                ..ConnectionSettings::default()
            };
            let Err(error) = settings.validate() else {
                panic!("server URL user-info must be rejected");
            };

            assert_eq!(error, "servers must not contain URL user-info");
            assert!(!error.contains(USER_PLACEHOLDER));
            assert!(!error.contains(PASSWORD_PLACEHOLDER));
        }
    }

    #[test]
    fn rejects_empty_and_malformed_servers() {
        for servers in ["", " , ", "not a url with spaces"] {
            let settings = ConnectionSettings {
                servers: servers.to_owned(),
                ..ConnectionSettings::default()
            };
            assert!(settings.validate().is_err());
        }
    }

    #[test]
    fn rejects_auth_conflict() {
        let settings = ConnectionSettings {
            credentials_file: Some("one".to_owned()),
            nkey_file: Some("two".to_owned()),
            ..ConnectionSettings::default()
        };
        assert!(settings.validate().is_err());
    }

    #[test]
    fn rejects_certificate_key_mismatch() {
        let settings = ConnectionSettings {
            tls_client_cert_file: Some("cert".to_owned()),
            ..ConnectionSettings::default()
        };
        assert!(settings.validate().is_err());
    }

    #[test]
    fn preserves_nanosecond_timeout() {
        let settings = ConnectionSettings {
            connection_timeout: 17,
            ..ConnectionSettings::default()
        };
        assert_eq!(
            settings.validate().expect("settings validate").timeout,
            Duration::from_nanos(17)
        );
    }
}
