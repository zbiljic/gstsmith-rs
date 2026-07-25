use std::sync::{LazyLock, Mutex, MutexGuard};

use futures_util::StreamExt;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::base_src::CreateSuccess;
use gst_base::subclass::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::connection::ConnectionSettings;
use crate::{message_meta, runtime};

const DEFAULT_SUBSCRIPTION_CAPACITY: u32 = 1024;

async fn wait_or_cancel<F, T>(cancel: &CancellationToken, future: F) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::select! {
        () = cancel.cancelled() => None,
        output = future => Some(output),
    }
}

#[derive(Clone)]
struct Settings {
    connection: ConnectionSettings,
    subject: String,
    queue_group: String,
    subscription_capacity: u32,
    caps: Option<gst::Caps>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            connection: ConnectionSettings::default(),
            subject: String::new(),
            queue_group: String::new(),
            subscription_capacity: DEFAULT_SUBSCRIPTION_CAPACITY,
            caps: None,
        }
    }
}

#[derive(Default)]
enum State {
    #[default]
    Stopped,
    Started {
        client: async_nats::Client,
        subscriber: Option<async_nats::Subscriber>,
        cancel: CancellationToken,
        timeout: std::time::Duration,
    },
}

#[derive(Default)]
pub struct NatsSrc {
    settings: Mutex<Settings>,
    state: Mutex<State>,
}

impl NatsSrc {
    fn settings(&self) -> MutexGuard<'_, Settings> {
        self.settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn settings_error(detail: impl std::fmt::Display) -> gst::ErrorMessage {
        gst::error_msg!(
            gst::ResourceError::Settings,
            ["Invalid NATS source settings: {detail}"]
        )
    }
}

#[glib::object_subclass]
impl ObjectSubclass for NatsSrc {
    const NAME: &'static str = "GstSmithNatsSrc";
    type Type = super::NatsSrc;
    type ParentType = gst_base::PushSrc;
}

impl ObjectImpl for NatsSrc {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_live(true);
        self.obj().set_format(gst::Format::Time);
        self.obj().set_do_timestamp(true);
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let mut properties = ConnectionSettings::property_specs();
            properties.extend([
                glib::ParamSpecString::builder("subject")
                    .nick("Subject")
                    .blurb("Core NATS subscription subject")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("queue-group")
                    .nick("Queue Group")
                    .blurb("Optional Core NATS queue group")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("subscription-capacity")
                    .nick("Subscription Capacity")
                    .blurb("Maximum pending messages in the NATS subscription")
                    .minimum(1)
                    .default_value(DEFAULT_SUBSCRIPTION_CAPACITY)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoxed::builder::<gst::Caps>("caps")
                    .nick("Caps")
                    .blurb("Optional fixed caps applied to source buffers")
                    .mutable_ready()
                    .build(),
            ]);
            properties
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings();
        if settings.connection.set_property(value, pspec) {
            return;
        }
        match pspec.name() {
            "subject" => {
                if let Ok(subject) = value.get::<String>() {
                    settings.subject = subject;
                }
            }
            "queue-group" => {
                if let Ok(queue_group) = value.get::<String>() {
                    settings.queue_group = queue_group;
                }
            }
            "subscription-capacity" => {
                if let Ok(capacity) = value.get::<u32>() {
                    settings.subscription_capacity = capacity;
                }
            }
            "caps" => {
                if let Ok(caps) = value.get::<Option<gst::Caps>>() {
                    settings.caps = caps;
                }
            }
            _ => {}
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings();
        if let Some(value) = settings.connection.property(pspec) {
            return value;
        }
        match pspec.name() {
            "subject" => settings.subject.to_value(),
            "queue-group" => settings.queue_group.to_value(),
            "subscription-capacity" => settings.subscription_capacity.to_value(),
            "caps" => settings.caps.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for NatsSrc {}

impl ElementImpl for NatsSrc {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Core NATS Source",
                "Source/Network",
                "Subscribes to Core NATS messages as arbitrary byte buffers",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });
        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "constant pad names and static caps make template construction infallible"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            vec![
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &gst::Caps::new_any(),
                )
                .expect("constructing the natssrc pad template"),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseSrcImpl for NatsSrc {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        if settings.subject.is_empty() {
            return Err(Self::settings_error("subject must not be empty"));
        }
        let validated = settings
            .connection
            .validate()
            .map_err(Self::settings_error)?;
        let capacity =
            usize::try_from(settings.subscription_capacity).map_err(Self::settings_error)?;
        let options = settings
            .connection
            .options(&validated, self.obj().name().as_str(), capacity)
            .map_err(Self::settings_error)?;
        let options = crate::connection::observe_events(options, self.obj().downgrade());
        let runtime = runtime::runtime().map_err(Self::settings_error)?;
        let subject = settings.subject.clone();
        let queue_group = settings.queue_group.clone();
        let retry_on_initial_connect = settings.connection.retry_on_initial_connect;
        let servers = validated.servers.clone();
        let (client, subscriber) = runtime.block_on(async move {
            let client = options.connect(servers).await.map_err(|_connect_error| {
                Self::settings_error("failed to connect to configured NATS servers")
            })?;
            let subscriber = if queue_group.is_empty() {
                client.subscribe(subject).await
            } else {
                client.queue_subscribe(subject, queue_group).await
            }
            .map_err(|_subscribe_error| {
                Self::settings_error("failed to create the NATS subscription")
            })?;
            if !retry_on_initial_connect {
                client.flush().await.map_err(|_flush_error| {
                    Self::settings_error("failed to activate the NATS subscription")
                })?;
            }
            Ok::<_, gst::ErrorMessage>((client, subscriber))
        })?;

        if let Some(caps) = settings.caps.as_ref() {
            self.obj()
                .set_caps(caps)
                .map_err(|error| Self::settings_error(format!("failed to set caps: {error}")))?;
        }
        *self.state() = State::Started {
            client,
            subscriber: Some(subscriber),
            cancel: CancellationToken::new(),
            timeout: validated.timeout,
        };
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let old_state = std::mem::take(&mut *self.state());
        if let State::Started {
            client,
            mut subscriber,
            cancel,
            timeout,
        } = old_state
        {
            cancel.cancel();
            let runtime = runtime::runtime().map_err(Self::settings_error)?;
            let _result = runtime.block_on(async move {
                tokio::time::timeout(timeout, async move {
                    if let Some(subscriber) = subscriber.as_mut() {
                        let _unsubscribe_result = subscriber.unsubscribe().await;
                    }
                    let _flush_result = client.flush().await;
                })
                .await
            });
        }
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn unlock(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { cancel, .. } = &*self.state() {
            cancel.cancel();
        }
        Ok(())
    }

    fn unlock_stop(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { cancel, .. } = &mut *self.state() {
            *cancel = CancellationToken::new();
        }
        Ok(())
    }
}

impl PushSrcImpl for NatsSrc {
    fn create(
        &self,
        _buffer: Option<&mut gst::BufferRef>,
    ) -> Result<CreateSuccess, gst::FlowError> {
        let (mut subscriber, cancel) = {
            let mut state = self.state();
            match &mut *state {
                State::Started {
                    subscriber, cancel, ..
                } => (
                    subscriber.take().ok_or(gst::FlowError::Error)?,
                    cancel.clone(),
                ),
                State::Stopped => return Err(gst::FlowError::Flushing),
            }
        };

        let runtime = runtime::runtime().map_err(|error| {
            gst::element_imp_error!(self, gst::ResourceError::Failed, ["{error}"]);
            gst::FlowError::Error
        })?;
        let next = runtime.block_on(wait_or_cancel(&cancel, subscriber.next()));

        let unclaimed_subscriber = {
            let mut state = self.state();
            if let State::Started {
                subscriber: slot, ..
            } = &mut *state
            {
                *slot = Some(subscriber);
                None
            } else {
                Some(subscriber)
            }
        };
        if let Some(mut subscriber) = unclaimed_subscriber {
            runtime.block_on(async move {
                let _unsubscribe_result = subscriber.unsubscribe().await;
            });
            return Err(gst::FlowError::Flushing);
        }

        let message = match next {
            None => return Err(gst::FlowError::Flushing),
            Some(Some(message)) => message,
            Some(None) => {
                gst::element_imp_error!(
                    self,
                    gst::ResourceError::Read,
                    ["The Core NATS subscription closed unexpectedly"]
                );
                return Err(gst::FlowError::Error);
            }
        };
        let mut buffer = gst::Buffer::from_mut_slice(message.payload.to_vec());
        message_meta::attach(
            buffer.get_mut().ok_or(gst::FlowError::Error)?,
            message.subject.as_str(),
            message.reply.as_ref().map(async_nats::Subject::as_str),
            message.headers.as_ref(),
        )
        .map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::StreamError::Format,
                ["Failed to attach NATS message metadata: {error}"]
            );
            gst::FlowError::Error
        })?;
        Ok(CreateSuccess::NewBuffer(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_wait_finishes_without_polling_the_message_future() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let started = std::time::Instant::now();
        let result = runtime::runtime()
            .expect("test runtime")
            .block_on(wait_or_cancel(
                &cancel,
                std::future::pending::<Option<async_nats::Message>>(),
            ));
        assert!(result.is_none());
        assert!(started.elapsed() < std::time::Duration::from_millis(100));
    }
}
