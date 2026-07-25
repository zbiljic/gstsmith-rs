use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;

use crate::connection::ConnectionSettings;
use crate::{message_meta, runtime};

const DEFAULT_QUEUE_CAPACITY: u32 = 64;
const DEFAULT_DRAIN_TIMEOUT: u64 = 2_000_000_000;

#[derive(Clone)]
struct Settings {
    connection: ConnectionSettings,
    subject: String,
    headers: gst::Array,
    queue_capacity: u32,
    drop_on_full: bool,
    drain_timeout: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            connection: ConnectionSettings::default(),
            subject: String::new(),
            headers: gst::Array::default(),
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            drop_on_full: false,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }
}

struct PublishRequest {
    subject: String,
    reply_subject: Option<String>,
    headers: Option<async_nats::HeaderMap>,
    payload: Vec<u8>,
}

#[derive(Default)]
enum State {
    #[default]
    Stopped,
    Started {
        sender: Option<tokio::sync::mpsc::Sender<PublishRequest>>,
        worker: tokio::task::JoinHandle<()>,
        failure: Arc<Mutex<Option<String>>>,
        error_posted: Arc<AtomicBool>,
        flushing: bool,
        fixed_headers: Option<async_nats::HeaderMap>,
        drain_timeout: std::time::Duration,
    },
}

#[derive(Default)]
pub struct NatsSink {
    settings: Mutex<Settings>,
    state: Mutex<State>,
    dropped_messages: AtomicU64,
}

impl NatsSink {
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
            ["Invalid NATS sink settings: {detail}"]
        )
    }

    fn render_error(&self, detail: impl std::fmt::Display) -> gst::FlowError {
        gst::element_imp_error!(
            self,
            gst::ResourceError::Write,
            ["Failed to publish a Core NATS message: {detail}"]
        );
        gst::FlowError::Error
    }

    fn render_worker_error(
        &self,
        detail: impl std::fmt::Display,
        error_posted: &AtomicBool,
    ) -> gst::FlowError {
        if !error_posted.swap(true, Ordering::Relaxed) {
            return self.render_error(detail);
        }
        gst::FlowError::Error
    }
}

#[glib::object_subclass]
impl ObjectSubclass for NatsSink {
    const NAME: &'static str = "GstSmithNatsSink";
    type Type = super::NatsSink;
    type ParentType = gst_base::BaseSink;
}

impl ObjectImpl for NatsSink {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_sync(false);
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let mut properties = ConnectionSettings::property_specs();
            properties.extend([
                glib::ParamSpecString::builder("subject")
                    .nick("Subject")
                    .blurb("Fixed publish subject; empty uses GstNatsMessageMeta")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                gst::ParamSpecArray::builder("headers")
                    .nick("Headers")
                    .blurb("Fixed NATS headers published before GstNatsMessageMeta headers")
                    .element_spec(
                        &glib::ParamSpecBoxed::builder::<gst::Structure>("nats-header")
                            .nick("NATS Header")
                            .blurb("Structure with string name and value fields")
                            .build(),
                    )
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("queue-capacity")
                    .nick("Queue Capacity")
                    .blurb("Maximum messages awaiting asynchronous publication")
                    .minimum(1)
                    .default_value(DEFAULT_QUEUE_CAPACITY)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("drop-on-full")
                    .nick("Drop On Full")
                    .blurb("Drop newest messages when the publication queue is full")
                    .default_value(false)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("drain-timeout")
                    .nick("Drain Timeout")
                    .blurb("Maximum stop drain time in nanoseconds; zero aborts immediately")
                    .default_value(DEFAULT_DRAIN_TIMEOUT)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("dropped-messages")
                    .nick("Dropped Messages")
                    .blurb("Messages dropped due to a full queue in the current run")
                    .read_only()
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
            "headers" => {
                if let Ok(headers) = value.get::<gst::Array>() {
                    settings.headers = headers;
                }
            }
            "queue-capacity" => {
                if let Ok(capacity) = value.get::<u32>() {
                    settings.queue_capacity = capacity;
                }
            }
            "drop-on-full" => {
                if let Ok(drop) = value.get::<bool>() {
                    settings.drop_on_full = drop;
                }
            }
            "drain-timeout" => {
                if let Ok(timeout) = value.get::<u64>() {
                    settings.drain_timeout = timeout;
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
            "headers" => settings.headers.to_value(),
            "queue-capacity" => settings.queue_capacity.to_value(),
            "drop-on-full" => settings.drop_on_full.to_value(),
            "drain-timeout" => settings.drain_timeout.to_value(),
            "dropped-messages" => self.dropped_messages.load(Ordering::Relaxed).to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for NatsSink {}

impl ElementImpl for NatsSink {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Core NATS Sink",
                "Sink/Network",
                "Publishes arbitrary byte buffers as Core NATS messages",
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
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &gst::Caps::new_any(),
                )
                .expect("constructing the natssink pad template"),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for NatsSink {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        let validated = settings
            .connection
            .validate()
            .map_err(Self::settings_error)?;
        if !settings.subject.is_empty()
            && settings
                .subject
                .bytes()
                .any(|byte| byte.is_ascii_whitespace())
        {
            return Err(Self::settings_error("subject contains whitespace"));
        }
        let fixed_headers =
            message_meta::headers_from_array(&settings.headers).map_err(Self::settings_error)?;
        let fixed_headers = (!fixed_headers.is_empty()).then_some(fixed_headers);
        let capacity = usize::try_from(settings.queue_capacity).map_err(Self::settings_error)?;
        let options = settings
            .connection
            .options(
                &validated,
                self.obj().name().as_str(),
                DEFAULT_QUEUE_CAPACITY as usize,
            )
            .map_err(Self::settings_error)?;
        let options = crate::connection::observe_events(options, self.obj().downgrade());
        let runtime = runtime::runtime().map_err(Self::settings_error)?;
        let servers = validated.servers;
        let client = runtime
            .block_on(options.connect(servers))
            .map_err(|_connect_error| {
                Self::settings_error("failed to connect to configured NATS servers")
            })?;
        let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
        let failure = Arc::new(Mutex::new(None));
        let error_posted = Arc::new(AtomicBool::new(false));
        let weak = self.obj().downgrade();
        let worker_failure = Arc::clone(&failure);
        let worker_error_posted = Arc::clone(&error_posted);
        let worker = runtime.spawn(publish_worker(
            client,
            receiver,
            worker_failure,
            worker_error_posted,
            weak,
        ));
        self.dropped_messages.store(0, Ordering::Relaxed);
        *self.state() = State::Started {
            sender: Some(sender),
            worker,
            failure,
            error_posted,
            flushing: false,
            fixed_headers,
            drain_timeout: std::time::Duration::from_nanos(settings.drain_timeout),
        };
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let old_state = std::mem::take(&mut *self.state());
        if let State::Started {
            sender,
            mut worker,
            drain_timeout,
            ..
        } = old_state
        {
            let queued_messages = sender
                .as_ref()
                .map_or(0, |sender| sender.max_capacity() - sender.capacity());
            drop(sender);
            if drain_timeout.is_zero() {
                worker.abort();
                let runtime = runtime::runtime().map_err(Self::settings_error)?;
                let _join_result = runtime.block_on(worker);
                if queued_messages > 0 {
                    gst::warning!(
                        gst::CAT_RUST,
                        imp = self,
                        "Abandoned {queued_messages} queued NATS messages during immediate shutdown"
                    );
                }
            } else {
                let runtime = runtime::runtime().map_err(Self::settings_error)?;
                let completed = runtime
                    .block_on(async { tokio::time::timeout(drain_timeout, &mut worker).await });
                if completed.is_err() {
                    worker.abort();
                    let _join_result = runtime.block_on(worker);
                    gst::warning!(
                        gst::CAT_RUST,
                        imp = self,
                        "Abandoned {queued_messages} queued NATS messages after drain timeout"
                    );
                }
            }
        }
        Ok(())
    }

    fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.settings().clone();
        let envelope = if message_meta::is_present(buffer) {
            Some(message_meta::read(buffer).map_err(|error| self.render_error(error))?)
        } else {
            None
        };
        let subject = if settings.subject.is_empty() {
            envelope
                .as_ref()
                .map(|envelope| envelope.subject.clone())
                .ok_or_else(|| {
                    self.render_error("no fixed subject or GstNatsMessageMeta subject")
                })?
        } else {
            settings.subject
        };
        let map = buffer
            .map_readable()
            .map_err(|error| self.render_error(format!("failed to map input buffer: {error}")))?;
        let reply_subject = envelope
            .as_ref()
            .and_then(|envelope| envelope.reply_subject.clone());
        let message_headers = envelope
            .as_ref()
            .and_then(|envelope| envelope.headers.as_ref());
        let payload = map.as_slice().to_vec();

        let state = self.state();
        let State::Started {
            sender,
            failure,
            error_posted,
            flushing,
            fixed_headers,
            ..
        } = &*state
        else {
            return Err(gst::FlowError::Flushing);
        };
        if *flushing {
            return Err(gst::FlowError::Flushing);
        }
        if let Some(error) = failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            return Err(self.render_worker_error(error, error_posted));
        }
        let sender = sender.as_ref().ok_or_else(|| {
            self.render_worker_error("publication worker is closed", error_posted)
        })?;
        let request = PublishRequest {
            subject,
            reply_subject,
            headers: message_meta::merge_headers(fixed_headers.as_ref(), message_headers),
            payload,
        };
        match sender.try_send(request) {
            Ok(()) => Ok(gst::FlowSuccess::Ok),
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) if settings.drop_on_full => {
                let dropped = self.dropped_messages.fetch_add(1, Ordering::Relaxed) + 1;
                drop(state);
                self.obj().notify("dropped-messages");
                if dropped.is_power_of_two() {
                    gst::warning!(
                        gst::CAT_RUST,
                        imp = self,
                        "Dropped {dropped} NATS messages because the queue was full"
                    );
                }
                Ok(gst::FlowSuccess::Ok)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                Err(self.render_error("publication queue is full"))
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                Err(self.render_worker_error("publication worker is closed", error_posted))
            }
        }
    }

    fn unlock(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { flushing, .. } = &mut *self.state() {
            *flushing = true;
        }
        Ok(())
    }

    fn unlock_stop(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { flushing, .. } = &mut *self.state() {
            *flushing = false;
        }
        Ok(())
    }
}

async fn publish_worker(
    client: async_nats::Client,
    mut receiver: tokio::sync::mpsc::Receiver<PublishRequest>,
    failure: Arc<Mutex<Option<String>>>,
    error_posted: Arc<AtomicBool>,
    weak: glib::WeakRef<super::NatsSink>,
) {
    while let Some(request) = receiver.recv().await {
        let payload = request.payload.into();
        let result = match (request.reply_subject, request.headers) {
            (None, None) => client.publish(request.subject, payload).await,
            (None, Some(headers)) => {
                client
                    .publish_with_headers(request.subject, headers, payload)
                    .await
            }
            (Some(reply), None) => {
                client
                    .publish_with_reply(request.subject, reply, payload)
                    .await
            }
            (Some(reply), Some(headers)) => {
                client
                    .publish_with_reply_and_headers(request.subject, reply, headers, payload)
                    .await
            }
        };
        if result.is_err() {
            let detail = "NATS client rejected a publish operation".to_owned();
            *failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(detail.clone());
            if !error_posted.swap(true, Ordering::Relaxed)
                && let Some(element) = weak.upgrade()
            {
                gst::element_error!(element, gst::ResourceError::Write, ["{detail}"]);
            }
            receiver.close();
            break;
        }
    }
    if client.flush().await.is_err() {
        let mut failure = failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if failure.is_none() {
            *failure = Some("NATS client flush failed".to_owned());
            drop(failure);
            if !error_posted.swap(true, Ordering::Relaxed)
                && let Some(element) = weak.upgrade()
            {
                gst::element_error!(
                    element,
                    gst::ResourceError::Write,
                    ["Core NATS client flush failed"]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        gst::init().expect("initializing GStreamer");
        crate::message_meta::register();
    }

    fn request() -> PublishRequest {
        PublishRequest {
            subject: "test.subject".to_owned(),
            reply_subject: None,
            headers: None,
            payload: vec![1],
        }
    }

    fn configured_sink(
        drop_on_full: bool,
    ) -> (
        super::super::NatsSink,
        tokio::sync::mpsc::Receiver<PublishRequest>,
    ) {
        init();
        let sink = glib::Object::builder::<super::super::NatsSink>().build();
        let imp = sink.imp();
        {
            let mut settings = imp.settings();
            settings.subject = "test.subject".to_owned();
            settings.drop_on_full = drop_on_full;
        }
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        sender.try_send(request()).expect("prefilling test queue");
        let worker = runtime::runtime()
            .expect("test runtime")
            .spawn(std::future::pending());
        *imp.state() = State::Started {
            sender: Some(sender),
            worker,
            failure: Arc::new(Mutex::new(None)),
            error_posted: Arc::new(AtomicBool::new(false)),
            flushing: false,
            fixed_headers: None,
            drain_timeout: std::time::Duration::ZERO,
        };
        (sink, receiver)
    }

    fn stop(sink: &super::super::NatsSink) {
        sink.imp().stop().expect("stopping test sink");
    }

    #[test]
    fn full_queue_is_an_error_by_default() {
        let (sink, _receiver) = configured_sink(false);
        assert_eq!(
            sink.imp().render(&gst::Buffer::from_slice([2])),
            Err(gst::FlowError::Error)
        );
        stop(&sink);
    }

    #[test]
    fn full_queue_can_drop_and_increment_the_counter() {
        let (sink, _receiver) = configured_sink(true);
        assert_eq!(
            sink.imp().render(&gst::Buffer::from_slice([2])),
            Ok(gst::FlowSuccess::Ok)
        );
        assert_eq!(sink.property::<u64>("dropped-messages"), 1);
        stop(&sink);
    }

    #[test]
    fn malformed_metadata_is_rejected_even_with_a_fixed_subject() {
        let (sink, _receiver) = configured_sink(false);
        let mut buffer = gst::Buffer::from_slice([2]);
        let mut meta = gst::meta::CustomMeta::add(
            buffer.get_mut().expect("new buffer is writable"),
            crate::message_meta::META_NAME,
        )
        .expect("adding test metadata");
        meta.mut_structure().set("subject", 42_i32);
        assert_eq!(sink.imp().render(&buffer), Err(gst::FlowError::Error));
        stop(&sink);
    }

    #[test]
    fn closed_worker_queue_is_reported_once_and_stop_is_bounded() {
        let (sink, receiver) = configured_sink(false);
        drop(receiver);
        assert_eq!(
            sink.imp().render(&gst::Buffer::from_slice([2])),
            Err(gst::FlowError::Error)
        );
        assert_eq!(
            sink.imp().render(&gst::Buffer::from_slice([3])),
            Err(gst::FlowError::Error)
        );
        let started = std::time::Instant::now();
        stop(&sink);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }
}
