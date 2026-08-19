use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use base64::Engine as _;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;

use crate::backend::{self, BackendError, GenerationRequest};
use crate::{prompt, runtime};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8000/v1/chat/completions";
const DEFAULT_USER_PROMPT: &str = "Describe what you see in these images.";
const DEFAULT_INTERVAL: u64 = 5_000_000_000;
const DEFAULT_MAX_TOKENS: u32 = 512;
const DEFAULT_TEMPERATURE: f64 = 0.2;
const DEFAULT_TOP_P: f64 = 0.9;
const DEFAULT_REQUEST_TIMEOUT: u64 = 30_000_000_000;
const DEFAULT_QUEUE_CAPACITY: u32 = 1;
const DEFAULT_MAX_FRAME_BYTES: u64 = 8_388_608;
const DEFAULT_MAX_BATCH_BYTES: u64 = 16_777_216;
const DEFAULT_DRAIN_TIMEOUT: u64 = 2_000_000_000;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "vlmanalysis",
        gst::DebugColorFlags::empty(),
        Some("Asynchronous VLM analysis"),
    )
});

#[derive(Clone)]
struct Settings {
    endpoint: String,
    allow_insecure_http: bool,
    api_key_file: Option<PathBuf>,
    model: String,
    system_prompt: Option<String>,
    user_prompt: String,
    analysis_interval: u64,
    frames_per_request: u32,
    max_tokens: u32,
    temperature: f64,
    top_p: f64,
    request_timeout: u64,
    queue_capacity: u32,
    max_frame_bytes: u64,
    max_batch_bytes: u64,
    drain_timeout: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_owned(),
            allow_insecure_http: false,
            api_key_file: None,
            model: String::new(),
            system_prompt: None,
            user_prompt: DEFAULT_USER_PROMPT.to_owned(),
            analysis_interval: DEFAULT_INTERVAL,
            frames_per_request: 1,
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_batch_bytes: DEFAULT_MAX_BATCH_BYTES,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    dropped: AtomicU64,
}

impl Counters {
    fn reset(&self) {
        self.submitted.store(0, Ordering::Relaxed);
        self.completed.store(0, Ordering::Relaxed);
        self.failed.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Sampler {
    last_seen_pts: Option<gst::ClockTime>,
    last_selected_pts: Option<gst::ClockTime>,
    no_pts_selected: bool,
    frames: Vec<SelectedFrame>,
    bytes: u64,
}

struct SelectedFrame {
    bytes: Vec<u8>,
    pts: Option<gst::ClockTime>,
}

struct Batch {
    id: u64,
    generation: u64,
    frames: Vec<SelectedFrame>,
    enqueued_at: Instant,
}

struct WorkerConfig {
    endpoint: reqwest::Url,
    api_key: Option<String>,
    model: String,
    system_prompt: Option<String>,
    user_prompt: String,
    max_tokens: u32,
    temperature: f64,
    top_p: f64,
    request_timeout: Duration,
}

#[derive(Default)]
enum State {
    #[default]
    Stopped,
    Started {
        sender: Option<tokio::sync::mpsc::Sender<Batch>>,
        worker: tokio::task::JoinHandle<()>,
        drain_timeout: Duration,
        error_posted: Arc<AtomicBool>,
    },
}

pub struct VlmAnalysis {
    settings: Mutex<Settings>,
    state: Mutex<State>,
    sampler: Mutex<Sampler>,
    counters: Arc<Counters>,
    generation: AtomicU64,
    next_id: AtomicU64,
    no_pts_warned: AtomicBool,
}

impl Default for VlmAnalysis {
    fn default() -> Self {
        Self {
            settings: Mutex::new(Settings::default()),
            state: Mutex::new(State::default()),
            sampler: Mutex::new(Sampler::default()),
            counters: Arc::new(Counters::default()),
            generation: AtomicU64::new(1),
            next_id: AtomicU64::new(1),
            no_pts_warned: AtomicBool::new(false),
        }
    }
}

impl VlmAnalysis {
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

    fn sampler(&self) -> MutexGuard<'_, Sampler> {
        self.sampler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn settings_error(detail: impl std::fmt::Display) -> gst::ErrorMessage {
        gst::error_msg!(
            gst::ResourceError::Settings,
            ["Invalid VLM analysis settings: {detail}"]
        )
    }

    fn reset_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        *self.sampler() = Sampler::default();
    }

    fn notify_counter(&self, name: &str) {
        self.obj().notify(name);
    }

    fn post_input_error(&self, message: &'static str) {
        self.counters.failed.fetch_add(1, Ordering::Relaxed);
        self.notify_counter("failed-requests");
        post_error_message(
            &self.obj(),
            None,
            self.generation.load(Ordering::Relaxed),
            "input",
            message,
            None,
            1,
        );
    }

    fn collect_batch(
        &self,
        buffer: &gst::BufferRef,
        settings: &Settings,
    ) -> Option<Vec<SelectedFrame>> {
        let pts = buffer.pts();
        let mut sampler = self.sampler();
        if let (Some(previous), Some(current)) = (sampler.last_seen_pts, pts)
            && current < previous
        {
            drop(sampler);
            self.reset_generation();
            sampler = self.sampler();
        }
        sampler.last_seen_pts = pts;
        if pts.is_none() && !self.no_pts_warned.swap(true, Ordering::Relaxed) {
            gst::warning!(
                CAT,
                imp = self,
                "Input has no PTS; only the first buffer is sampled unless analysis-interval is zero"
            );
        }
        let selected = match pts {
            Some(current) => sampler.last_selected_pts.is_none_or(|previous| {
                settings.analysis_interval == 0
                    || current.saturating_sub(previous).nseconds() >= settings.analysis_interval
            }),
            None => {
                settings.analysis_interval == 0
                    || !sampler.no_pts_selected && sampler.frames.is_empty()
            }
        };
        if !selected {
            return None;
        }
        let map = match buffer.map_readable() {
            Ok(map) => map,
            Err(_error) => {
                drop(sampler);
                self.post_input_error("failed to read selected JPEG buffer");
                return None;
            }
        };
        let Ok(size) = u64::try_from(map.size()) else {
            drop(map);
            drop(sampler);
            self.post_input_error("selected JPEG size does not fit the configured limit type");
            return None;
        };
        if size > settings.max_frame_bytes {
            drop(map);
            drop(sampler);
            self.post_input_error("selected JPEG exceeds max-frame-bytes");
            return None;
        }
        let next_bytes = sampler.bytes.checked_add(size);
        if next_bytes.is_none_or(|bytes| bytes > settings.max_batch_bytes) {
            drop(map);
            drop(sampler);
            self.post_input_error("selected JPEG would exceed max-batch-bytes");
            return None;
        }
        let bytes = map.as_slice().to_vec();
        drop(map);
        sampler.bytes = next_bytes.unwrap_or(settings.max_batch_bytes);
        if pts.is_none() {
            sampler.no_pts_selected = true;
        }
        sampler.last_selected_pts = pts.or(sampler.last_selected_pts);
        sampler.frames.push(SelectedFrame { bytes, pts });
        if sampler.frames.len() < usize::try_from(settings.frames_per_request).unwrap_or(usize::MAX)
        {
            return None;
        }
        let frames = std::mem::take(&mut sampler.frames);
        sampler.bytes = 0;
        Some(frames)
    }

    fn enqueue_batch(
        &self,
        frames: Vec<SelectedFrame>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let generation = self.generation.load(Ordering::Relaxed);
        let batch = Batch {
            id,
            generation,
            frames,
            enqueued_at: Instant::now(),
        };
        let frame_count = u32::try_from(batch.frames.len()).unwrap_or(u32::MAX);
        let state = self.state();
        let State::Started {
            sender,
            error_posted,
            ..
        } = &*state
        else {
            return Err(gst::FlowError::Flushing);
        };
        let Some(sender) = sender.as_ref() else {
            return Err(gst::FlowError::Flushing);
        };
        match sender.try_send(batch) {
            Ok(()) => {
                self.counters.submitted.fetch_add(1, Ordering::Relaxed);
                drop(state);
                self.notify_counter("submitted-requests");
                Ok(gst::FlowSuccess::Ok)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(batch)) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
                drop(state);
                self.notify_counter("dropped-batches");
                post_error_message(
                    &self.obj(),
                    Some(batch.id),
                    batch.generation,
                    "backpressure",
                    "VLM request queue is full; newest batch was dropped",
                    None,
                    frame_count,
                );
                Ok(gst::FlowSuccess::Ok)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_batch)) => {
                let error_posted = Arc::clone(error_posted);
                drop(state);
                if !error_posted.swap(true, Ordering::Relaxed) {
                    gst::element_imp_error!(
                        self,
                        gst::ResourceError::Failed,
                        ["VLM analysis worker is unavailable"]
                    );
                }
                Err(gst::FlowError::Flushing)
            }
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for VlmAnalysis {
    const NAME: &'static str = "GstSmithVlmAnalysis";
    type Type = super::VlmAnalysis;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for VlmAnalysis {
    #[expect(
        clippy::too_many_lines,
        reason = "keeping the complete public GObject property contract in one table makes inspection reliable"
    )]
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("endpoint")
                    .nick("Endpoint")
                    .blurb("Full OpenAI-compatible Chat Completions URL")
                    .default_value(Some(DEFAULT_ENDPOINT))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("allow-insecure-http")
                    .nick("Allow Insecure HTTP")
                    .blurb("Permit non-loopback plaintext HTTP, exposing images and credentials in transit")
                    .default_value(false)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("api-key-file")
                    .nick("API Key File")
                    .blurb("UTF-8 file containing a Bearer credential, read once at start")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("model")
                    .nick("Model")
                    .blurb("Required model identifier")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("system-prompt")
                    .nick("System Prompt")
                    .blurb("Optional literal system prompt")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("user-prompt")
                    .nick("User Prompt")
                    .blurb("Literal user prompt placed before image parts")
                    .default_value(Some(DEFAULT_USER_PROMPT))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("analysis-interval")
                    .nick("Analysis Interval")
                    .blurb("Minimum PTS spacing in nanoseconds; zero selects every buffer")
                    .default_value(DEFAULT_INTERVAL)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("frames-per-request")
                    .nick("Frames Per Request")
                    .blurb("Selected JPEG frames in each request")
                    .minimum(1)
                    .maximum(10)
                    .default_value(1)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-tokens")
                    .nick("Maximum Tokens")
                    .blurb("Maximum completion token count")
                    .minimum(1)
                    .default_value(DEFAULT_MAX_TOKENS)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecDouble::builder("temperature")
                    .nick("Temperature")
                    .blurb("Generation sampling temperature")
                    .minimum(0.0)
                    .maximum(2.0)
                    .default_value(DEFAULT_TEMPERATURE)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecDouble::builder("top-p")
                    .nick("Top P")
                    .blurb("Generation nucleus probability")
                    .minimum(f64::EPSILON)
                    .maximum(1.0)
                    .default_value(DEFAULT_TOP_P)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("request-timeout")
                    .nick("Request Timeout")
                    .blurb("HTTP send and response-read deadline in nanoseconds")
                    .minimum(1)
                    .default_value(DEFAULT_REQUEST_TIMEOUT)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("queue-capacity")
                    .nick("Queue Capacity")
                    .blurb("Complete batches waiting for the ordered worker")
                    .minimum(1)
                    .maximum(4)
                    .default_value(DEFAULT_QUEUE_CAPACITY)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("max-frame-bytes")
                    .nick("Maximum Frame Bytes")
                    .blurb("Maximum selected JPEG size")
                    .minimum(1)
                    .maximum(16 * 1024 * 1024)
                    .default_value(DEFAULT_MAX_FRAME_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("max-batch-bytes")
                    .nick("Maximum Batch Bytes")
                    .blurb("Maximum sum of selected JPEG bytes")
                    .minimum(1)
                    .maximum(32 * 1024 * 1024)
                    .default_value(DEFAULT_MAX_BATCH_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("drain-timeout")
                    .nick("Drain Timeout")
                    .blurb("Maximum stop drain time in nanoseconds; zero aborts immediately")
                    .default_value(DEFAULT_DRAIN_TIMEOUT)
                    .mutable_ready()
                    .build(),
                counter_property("submitted-requests", "Submitted Requests", "Complete batches queued in the current run"),
                counter_property("completed-requests", "Completed Requests", "Successful parsed responses in the current run"),
                counter_property("failed-requests", "Failed Requests", "Recoverable input or request failures in the current run"),
                counter_property("dropped-batches", "Dropped Batches", "Complete batches dropped because the queue was full"),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings();
        match pspec.name() {
            "endpoint" => set_string(value, &mut settings.endpoint),
            "allow-insecure-http" => set_copy(value, &mut settings.allow_insecure_http),
            "api-key-file" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    settings.api_key_file = path.map(PathBuf::from);
                }
            }
            "model" => set_string(value, &mut settings.model),
            "system-prompt" => {
                if let Ok(prompt) = value.get::<Option<String>>() {
                    settings.system_prompt = prompt;
                }
            }
            "user-prompt" => set_string(value, &mut settings.user_prompt),
            "analysis-interval" => set_copy(value, &mut settings.analysis_interval),
            "frames-per-request" => set_copy(value, &mut settings.frames_per_request),
            "max-tokens" => set_copy(value, &mut settings.max_tokens),
            "temperature" => set_copy(value, &mut settings.temperature),
            "top-p" => set_copy(value, &mut settings.top_p),
            "request-timeout" => set_copy(value, &mut settings.request_timeout),
            "queue-capacity" => set_copy(value, &mut settings.queue_capacity),
            "max-frame-bytes" => set_copy(value, &mut settings.max_frame_bytes),
            "max-batch-bytes" => set_copy(value, &mut settings.max_batch_bytes),
            "drain-timeout" => set_copy(value, &mut settings.drain_timeout),
            _ => gst::warning!(CAT, imp = self, "unexpected property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings();
        match pspec.name() {
            "endpoint" => settings.endpoint.to_value(),
            "allow-insecure-http" => settings.allow_insecure_http.to_value(),
            "api-key-file" => settings
                .api_key_file
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .to_value(),
            "model" => settings.model.to_value(),
            "system-prompt" => settings.system_prompt.to_value(),
            "user-prompt" => settings.user_prompt.to_value(),
            "analysis-interval" => settings.analysis_interval.to_value(),
            "frames-per-request" => settings.frames_per_request.to_value(),
            "max-tokens" => settings.max_tokens.to_value(),
            "temperature" => settings.temperature.to_value(),
            "top-p" => settings.top_p.to_value(),
            "request-timeout" => settings.request_timeout.to_value(),
            "queue-capacity" => settings.queue_capacity.to_value(),
            "max-frame-bytes" => settings.max_frame_bytes.to_value(),
            "max-batch-bytes" => settings.max_batch_bytes.to_value(),
            "drain-timeout" => settings.drain_timeout.to_value(),
            "submitted-requests" => self.counters.submitted.load(Ordering::Relaxed).to_value(),
            "completed-requests" => self.counters.completed.load(Ordering::Relaxed).to_value(),
            "failed-requests" => self.counters.failed.load(Ordering::Relaxed).to_value(),
            "dropped-batches" => self.counters.dropped.load(Ordering::Relaxed).to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

fn counter_property(name: &str, nick: &str, blurb: &str) -> glib::ParamSpec {
    glib::ParamSpecUInt64::builder(name)
        .nick(nick)
        .blurb(blurb)
        .read_only()
        .build()
}

fn set_copy<T: for<'a> glib::value::FromValue<'a> + Copy>(value: &glib::Value, target: &mut T) {
    if let Ok(new_value) = value.get::<T>() {
        *target = new_value;
    }
}

fn set_string(value: &glib::Value, target: &mut String) {
    if let Ok(new_value) = value.get::<String>() {
        *target = new_value;
    }
}

impl GstObjectImpl for VlmAnalysis {}

impl ElementImpl for VlmAnalysis {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Vision-Language Model Analysis",
                "Filter/Analysis/Video",
                "Samples JPEG frames for asynchronous OpenAI-compatible VLM analysis",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });
        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "static JPEG pad templates are infallible"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = gst::Caps::builder("image/jpeg").build();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .expect("constructing vlmanalysis sink template"),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .expect("constructing vlmanalysis src template"),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for VlmAnalysis {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        let config = validate_settings(&settings).map_err(Self::settings_error)?;
        ensure_crypto_provider().map_err(Self::settings_error)?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_error| Self::settings_error("failed to build the HTTP client"))?;
        let capacity = usize::try_from(settings.queue_capacity).map_err(Self::settings_error)?;
        let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
        let counters = Arc::clone(&self.counters);
        let weak = self.obj().downgrade();
        let runtime = runtime::runtime().map_err(Self::settings_error)?;
        let worker = runtime.spawn(worker_loop(client, config, receiver, counters, weak));
        self.counters.reset();
        self.generation.store(1, Ordering::Relaxed);
        self.next_id.store(1, Ordering::Relaxed);
        self.no_pts_warned.store(false, Ordering::Relaxed);
        *self.sampler() = Sampler::default();
        *self.state() = State::Started {
            sender: Some(sender),
            worker,
            drain_timeout: Duration::from_nanos(settings.drain_timeout),
            error_posted: Arc::new(AtomicBool::new(false)),
        };
        for name in [
            "submitted-requests",
            "completed-requests",
            "failed-requests",
            "dropped-batches",
        ] {
            self.notify_counter(name);
        }
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let old_state = std::mem::take(&mut *self.state());
        *self.sampler() = Sampler::default();
        if let State::Started {
            sender,
            mut worker,
            drain_timeout,
            ..
        } = old_state
        {
            let queued = sender.as_ref().map_or(0, |sender| {
                sender.max_capacity().saturating_sub(sender.capacity())
            });
            drop(sender);
            let runtime = runtime::runtime().map_err(Self::settings_error)?;
            if drain_timeout.is_zero() {
                worker.abort();
                let _join_result = runtime.block_on(worker);
                if queued > 0 {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "Abandoned {queued} queued VLM batches during immediate shutdown"
                    );
                }
            } else if runtime
                .block_on(async { tokio::time::timeout(drain_timeout, &mut worker).await })
                .is_err()
            {
                worker.abort();
                let _join_result = runtime.block_on(worker);
                gst::warning!(
                    CAT,
                    imp = self,
                    "Abandoned {queued} queued VLM batches after drain timeout"
                );
            }
        }
        Ok(())
    }

    fn sink_event(&self, event: gst::Event) -> bool {
        if matches!(
            event.view(),
            gst::EventView::StreamStart(_)
                | gst::EventView::Segment(_)
                | gst::EventView::FlushStop(_)
        ) {
            self.reset_generation();
        }
        self.parent_sink_event(event)
    }

    fn transform_ip(
        &self,
        buffer: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.settings().clone();
        self.collect_batch(buffer, &settings)
            .map_or(Ok(gst::FlowSuccess::Ok), |frames| {
                self.enqueue_batch(frames)
            })
    }
}

fn validate_settings(settings: &Settings) -> Result<WorkerConfig, &'static str> {
    if settings.model.is_empty() {
        return Err("model must be non-empty");
    }
    let endpoint =
        reqwest::Url::parse(&settings.endpoint).map_err(|_error| "endpoint is not a valid URL")?;
    if !matches!(endpoint.scheme(), "http" | "https") {
        return Err("endpoint scheme must be HTTP or HTTPS");
    }
    if !endpoint.username().is_empty() || endpoint.password().is_some() {
        return Err("endpoint must not contain user information");
    }
    if endpoint.fragment().is_some() {
        return Err("endpoint must not contain a fragment");
    }
    if endpoint.scheme() == "http" && !settings.allow_insecure_http && !is_loopback_host(&endpoint)
    {
        return Err("non-loopback plaintext HTTP requires allow-insecure-http=true");
    }
    if !(1..=10).contains(&settings.frames_per_request) {
        return Err("frames-per-request must be from 1 through 10");
    }
    if settings.max_tokens == 0 {
        return Err("max-tokens must be nonzero");
    }
    if !settings.temperature.is_finite() || !(0.0..=2.0).contains(&settings.temperature) {
        return Err("temperature must be finite and from 0.0 through 2.0");
    }
    if !settings.top_p.is_finite() || settings.top_p <= 0.0 || settings.top_p > 1.0 {
        return Err("top-p must be finite, greater than 0.0, and at most 1.0");
    }
    if settings.request_timeout == 0 {
        return Err("request-timeout must be nonzero");
    }
    if !(1..=4).contains(&settings.queue_capacity) {
        return Err("queue-capacity must be from 1 through 4");
    }
    if !(1..=16 * 1024 * 1024).contains(&settings.max_frame_bytes) {
        return Err("max-frame-bytes must be from 1 through 16 MiB");
    }
    if !(1..=32 * 1024 * 1024).contains(&settings.max_batch_bytes) {
        return Err("max-batch-bytes must be from 1 through 32 MiB");
    }
    let maximum = u64::from(settings.frames_per_request)
        .checked_mul(settings.max_frame_bytes)
        .ok_or("frame limits overflow")?;
    if maximum > settings.max_batch_bytes {
        return Err("frames-per-request * max-frame-bytes exceeds max-batch-bytes");
    }
    let api_key = settings
        .api_key_file
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|_error| "failed to read api-key-file as UTF-8")
        })
        .transpose()?
        .map(|key| key.trim_end().to_owned());
    if api_key.as_ref().is_some_and(String::is_empty) {
        return Err("api-key-file is empty after trimming trailing whitespace");
    }
    Ok(WorkerConfig {
        endpoint,
        api_key,
        model: settings.model.clone(),
        system_prompt: settings.system_prompt.clone(),
        user_prompt: settings.user_prompt.clone(),
        max_tokens: settings.max_tokens,
        temperature: settings.temperature,
        top_p: settings.top_p,
        request_timeout: Duration::from_nanos(settings.request_timeout),
    })
}

fn ensure_crypto_provider() -> Result<(), &'static str> {
    if rustls::crypto::CryptoProvider::get_default().is_none()
        && rustls::crypto::ring::default_provider()
            .install_default()
            .is_err()
        && rustls::crypto::CryptoProvider::get_default().is_none()
    {
        return Err("failed to install the TLS crypto provider");
    }
    Ok(())
}

fn is_loopback_host(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|host| {
        let host = host
            .strip_prefix('[')
            .and_then(|host| host.strip_suffix(']'))
            .unwrap_or(host);
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

async fn worker_loop(
    client: reqwest::Client,
    config: WorkerConfig,
    mut receiver: tokio::sync::mpsc::Receiver<Batch>,
    counters: Arc<Counters>,
    weak: glib::WeakRef<super::VlmAnalysis>,
) {
    while let Some(batch) = receiver.recv().await {
        let frame_count = u32::try_from(batch.frames.len()).unwrap_or(u32::MAX);
        let (start_pts, end_pts) = batch_pts(&batch.frames);
        let data_urls = batch
            .frames
            .into_iter()
            .map(|frame| {
                let encoded = base64::engine::general_purpose::STANDARD.encode(frame.bytes);
                format!("data:image/jpeg;base64,{encoded}")
            })
            .collect();
        let messages = prompt::literal_messages(
            config.system_prompt.clone(),
            config.user_prompt.clone(),
            data_urls,
        );
        let request = GenerationRequest {
            model: config.model.clone(),
            messages,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
        };
        let outcome = backend::generate(
            &client,
            config.endpoint.clone(),
            config.api_key.as_deref(),
            request,
            config.request_timeout,
        )
        .await;
        let latency = u64::try_from(batch.enqueued_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let Some(element) = weak.upgrade() else {
            break;
        };
        match outcome {
            Ok(result) => {
                counters.completed.fetch_add(1, Ordering::Relaxed);
                element.notify("completed-requests");
                post_result_message(
                    &element,
                    batch.id,
                    batch.generation,
                    &config.model,
                    &result.text,
                    frame_count,
                    start_pts,
                    end_pts,
                    latency,
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                );
            }
            Err(error) => {
                counters.failed.fetch_add(1, Ordering::Relaxed);
                element.notify("failed-requests");
                let (kind, message, status) = match error {
                    BackendError::Timeout => ("timeout", "VLM request timed out", None),
                    BackendError::Http {
                        status,
                        body_bytes,
                        message,
                    } => {
                        if let Some(body_bytes) = body_bytes {
                            gst::debug!(
                                CAT,
                                obj = &element,
                                "Discarded {body_bytes} provider response body bytes"
                            );
                        }
                        ("http", message, status.map(u32::from))
                    }
                    BackendError::Response(message) => ("response", message, None),
                };
                post_error_message(
                    &element,
                    Some(batch.id),
                    batch.generation,
                    kind,
                    message,
                    status,
                    frame_count,
                );
            }
        }
    }
}

fn batch_pts(frames: &[SelectedFrame]) -> (Option<gst::ClockTime>, Option<gst::ClockTime>) {
    let mut valid = frames.iter().filter_map(|frame| frame.pts);
    let first = valid.next();
    let last = valid.next_back().or(first);
    (first, last)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the public message schema has fixed fields"
)]
fn post_result_message(
    element: &super::VlmAnalysis,
    id: u64,
    generation: u64,
    model: &str,
    text: &str,
    frame_count: u32,
    start_pts: Option<gst::ClockTime>,
    end_pts: Option<gst::ClockTime>,
    latency: u64,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
) {
    let mut structure = gst::Structure::builder("vlmanalysis-result")
        .field("request-id", id)
        .field("generation", generation)
        .field("model", model)
        .field("text", text)
        .field("frame-count", frame_count)
        .field("latency", latency)
        .build();
    if let Some(pts) = start_pts {
        structure.set("start-pts", pts);
    }
    if let Some(pts) = end_pts {
        structure.set("end-pts", pts);
    }
    if let Some(tokens) = prompt_tokens {
        structure.set("prompt-tokens", tokens);
    }
    if let Some(tokens) = completion_tokens {
        structure.set("completion-tokens", tokens);
    }
    let message = gst::message::Element::builder(structure)
        .src(element)
        .build();
    let _post_result = element.post_message(message);
}

fn post_error_message(
    element: &super::VlmAnalysis,
    id: Option<u64>,
    generation: u64,
    kind: &str,
    message: &str,
    http_status: Option<u32>,
    frame_count: u32,
) {
    let mut structure = gst::Structure::builder("vlmanalysis-error")
        .field("generation", generation)
        .field("kind", kind)
        .field("message", message)
        .field("frame-count", frame_count)
        .build();
    if let Some(id) = id {
        structure.set("request-id", id);
    }
    if let Some(status) = http_status {
        structure.set("http-status", status);
    }
    let gst_message = gst::message::Element::builder(structure)
        .src(element)
        .build();
    let _post_result = element.post_message(gst_message);
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;

    fn init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            gst::init().expect("initializing GStreamer for worker test");
        });
    }

    #[test]
    fn worker_disappearance_posts_one_fatal_error_and_flushes() {
        init();
        let element: super::super::VlmAnalysis = glib::Object::builder()
            .property("model", "test-model")
            .build();
        let bus = gst::Bus::new();
        element.set_bus(Some(&bus));
        let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
        harness.set_src_caps_str("image/jpeg");
        harness.play();
        let (sender, abort) = {
            let state = element.imp().state();
            assert!(matches!(
                &*state,
                State::Started {
                    sender: Some(_),
                    ..
                }
            ));
            match &*state {
                State::Started {
                    sender: Some(sender),
                    worker,
                    ..
                } => (sender.clone(), worker.abort_handle()),
                _ => return,
            }
        };
        abort.abort();
        runtime::runtime()
            .expect("accessing VLM runtime")
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(1), sender.closed())
                    .await
                    .expect("worker receiver closes after abort");
            });
        assert_eq!(
            harness.push(gst::Buffer::from_mut_slice(b"jpeg".to_vec())),
            Err(gst::FlowError::Flushing)
        );
        let message = bus
            .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Error])
            .expect("receiving fatal worker error");
        assert!(matches!(message.view(), gst::MessageView::Error(_)));
        assert!(
            bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Error])
                .is_none()
        );
    }
}
