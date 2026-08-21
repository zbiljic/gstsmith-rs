use std::net::SocketAddr;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use parking_lot::Mutex;
use regex::Regex;

use crate::metrics::{Metrics, MetricsSlot};
use crate::worker::{self, WorkerConfig, WorkerHandle};

const DEFAULT_DESTINATION: &str = "127.0.0.1:8125";
const DEFAULT_PREFIX: &str = "gstsmith";

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "statsd",
        gst::DebugColorFlags::empty(),
        Some("StatsD metrics tracer"),
    )
});

static QUEUE_TYPE: LazyLock<glib::Type> = LazyLock::new(|| queue_type("queue"));
static QUEUE2_TYPE: LazyLock<glib::Type> = LazyLock::new(|| queue_type("queue2"));

fn queue_type(factory_name: &str) -> glib::Type {
    let Some(factory) = gst::ElementFactory::find(factory_name) else {
        gst::warning!(CAT, "GStreamer queue factory {factory_name} is unavailable");
        return glib::Type::INVALID;
    };
    match factory.load() {
        Ok(factory) => factory.element_type(),
        Err(_error) => {
            gst::warning!(
                CAT,
                "GStreamer queue factory {factory_name} could not be loaded"
            );
            glib::Type::INVALID
        }
    }
}

fn is_supported_queue(element: &gst::Element) -> bool {
    element.type_() == *QUEUE_TYPE || element.type_() == *QUEUE2_TYPE
}

#[derive(Clone)]
struct Settings {
    destination: String,
    prefix: String,
    global_tags: Option<String>,
    flush_interval_ms: u32,
    include_filter: Option<String>,
    exclude_filter: Option<String>,
    max_pad_series: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            destination: DEFAULT_DESTINATION.to_owned(),
            prefix: DEFAULT_PREFIX.to_owned(),
            global_tags: None,
            flush_interval_ms: 1_000,
            include_filter: None,
            exclude_filter: None,
            max_pad_series: 256,
        }
    }
}

#[derive(Default)]
struct RuntimeState {
    worker: Option<WorkerHandle>,
}

#[derive(Default)]
pub struct StatsdTracer {
    settings: Mutex<Settings>,
    // gstreamer-rs applies GST_TRACERS parameters with ordinary set_property() calls from its
    // constructed() wrapper, which rejects CONSTRUCT_ONLY properties. These use CONSTRUCT for
    // compatibility; this flag preserves configuration-at-creation semantics after construction.
    settings_frozen: std::sync::atomic::AtomicBool,
    runtime: Mutex<RuntimeState>,
    metrics: MetricsSlot,
}

#[glib::object_subclass]
impl ObjectSubclass for StatsdTracer {
    const NAME: &'static str = "GstSmithStatsdTracer";
    type Type = super::StatsdTracer;
    type ParentType = gst::Tracer;
}

impl ObjectImpl for StatsdTracer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("destination")
                    .nick("Destination")
                    .blurb("Numeric UDP destination for DogStatsD metrics")
                    .default_value(Some(DEFAULT_DESTINATION))
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("prefix")
                    .nick("Metric prefix")
                    .blurb("ASCII prefix prepended to every metric key")
                    .default_value(Some(DEFAULT_PREFIX))
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("global-tags")
                    .nick("Global tags")
                    .blurb("Comma-separated key:value DogStatsD tags")
                    .construct()
                    .build(),
                glib::ParamSpecUInt::builder("flush-interval-ms")
                    .nick("Flush interval")
                    .blurb("StatsD worker flush interval in milliseconds")
                    .minimum(100)
                    .maximum(60_000)
                    .default_value(1_000)
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("include-filter")
                    .nick("Include filter")
                    .blurb("Optional regular expression selecting raw metric scopes")
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("exclude-filter")
                    .nick("Exclude filter")
                    .blurb("Optional regular expression excluding raw metric scopes")
                    .construct()
                    .build(),
                glib::ParamSpecUInt::builder("max-pad-series")
                    .nick("Maximum pad series")
                    .blurb("Maximum number of active pad label sets")
                    .minimum(1)
                    .maximum(65_535)
                    .default_value(256)
                    .construct()
                    .build(),
                glib::ParamSpecBoolean::builder("worker-running")
                    .nick("Worker running")
                    .blurb("Whether the StatsD exporter worker started successfully")
                    .default_value(false)
                    .read_only()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        if self
            .settings_frozen
            .load(std::sync::atomic::Ordering::Acquire)
        {
            gst::warning!(
                CAT,
                imp = self,
                "Ignoring change to configuration property {} after tracer creation",
                pspec.name()
            );
            return;
        }
        let mut settings = self.settings.lock();
        match pspec.name() {
            "destination" => set_string(value, &mut settings.destination),
            "prefix" => set_string(value, &mut settings.prefix),
            "global-tags" => set_optional_string(value, &mut settings.global_tags),
            "flush-interval-ms" => set_u32(value, &mut settings.flush_interval_ms),
            "include-filter" => set_optional_string(value, &mut settings.include_filter),
            "exclude-filter" => set_optional_string(value, &mut settings.exclude_filter),
            "max-pad-series" => set_u32(value, &mut settings.max_pad_series),
            _ => gst::warning!(CAT, imp = self, "Unknown property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "destination" => self.settings.lock().destination.to_value(),
            "prefix" => self.settings.lock().prefix.to_value(),
            "global-tags" => self.settings.lock().global_tags.to_value(),
            "flush-interval-ms" => self.settings.lock().flush_interval_ms.to_value(),
            "include-filter" => self.settings.lock().include_filter.to_value(),
            "exclude-filter" => self.settings.lock().exclude_filter.to_value(),
            "max-pad-series" => self.settings.lock().max_pad_series.to_value(),
            "worker-running" => self
                .runtime
                .lock()
                .worker
                .as_ref()
                .is_some_and(WorkerHandle::is_running)
                .to_value(),
            _ => pspec.default_value().clone(),
        }
    }

    fn constructed(&self) {
        self.settings_frozen
            .store(true, std::sync::atomic::Ordering::Release);
        self.parent_constructed();

        let settings = self.settings.lock().clone();
        let config = match validate_settings(&settings) {
            Ok(config) => config,
            Err(error) => {
                gst::error!(CAT, imp = self, "{error}");
                return;
            }
        };
        let include = match compile_filter("include-filter", settings.include_filter.as_deref()) {
            Ok(filter) => filter,
            Err(error) => {
                gst::error!(CAT, imp = self, "{error}");
                return;
            }
        };
        let exclude = match compile_filter("exclude-filter", settings.exclude_filter.as_deref()) {
            Ok(filter) => filter,
            Err(error) => {
                gst::error!(CAT, imp = self, "{error}");
                return;
            }
        };
        let (metrics, retired) = Metrics::new(include, exclude, settings.max_pad_series as usize);
        let worker = match worker::start(config, Arc::clone(&metrics), retired) {
            Ok(worker) => worker,
            Err(error) => {
                gst::error!(CAT, imp = self, "Failed to start StatsD exporter: {error}");
                return;
            }
        };
        if !self.metrics.install(metrics) {
            let mut worker = worker;
            if let Err(error) = worker.stop() {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to stop duplicate StatsD worker: {error}"
                );
            }
            gst::error!(CAT, imp = self, "StatsD metrics were already initialized");
            return;
        }
        self.runtime.lock().worker = Some(worker);
        self.obj().notify("worker-running");

        LazyLock::force(&QUEUE_TYPE);
        LazyLock::force(&QUEUE2_TYPE);
        self.register_hook(TracerHook::ElementNew);
        self.register_hook(TracerHook::ElementAddPad);
        self.register_hook(TracerHook::ElementRemovePad);
        self.register_hook(TracerHook::BinAddPost);
        self.register_hook(TracerHook::BinRemovePre);
        self.register_hook(TracerHook::ObjectDestroyed);
        self.register_hook(TracerHook::ElementChangeStatePost);
        self.register_hook(TracerHook::PadPushPre);
        self.register_hook(TracerHook::PadPushListPre);
    }

    fn dispose(&self) {
        let mut worker = self.runtime.lock().worker.take();
        if let Some(worker) = worker.as_mut()
            && let Err(error) = worker.stop()
        {
            gst::error!(CAT, imp = self, "Failed to stop StatsD worker: {error}");
        }
        self.obj().notify("worker-running");
    }
}
impl GstObjectImpl for StatsdTracer {}
impl TracerImpl for StatsdTracer {
    const USE_STRUCTURE_PARAMS: bool = true;

    fn element_new(&self, _ts: u64, element: &gst::Element) {
        if let Some(pipeline) = element.downcast_ref::<gst::Pipeline>()
            && let Some(metrics) = self.metrics.get()
        {
            metrics.track_pipeline(pipeline);
        }
    }

    fn element_add_pad(&self, _ts: u64, _element: &gst::Element, _pad: &gst::Pad) {}

    fn element_remove_pad(&self, _ts: u64, _element: &gst::Element, pad: &gst::Pad) {
        if let Some(metrics) = self.metrics.get() {
            metrics.remove_pad(pad);
        }
    }

    fn bin_add_post(&self, _ts: u64, _bin: &gst::Bin, element: &gst::Element, success: bool) {
        if success {
            self.track_element(element);
        }
    }

    fn bin_remove_pre(&self, _ts: u64, _bin: &gst::Bin, element: &gst::Element) {
        if let Some(metrics) = self.metrics.get() {
            metrics.remove_object_key(element.as_ptr() as usize);
        }
    }

    fn object_destroyed(&self, _ts: u64, object: std::ptr::NonNull<gst::ffi::GstObject>) {
        if let Some(metrics) = self.metrics.get() {
            metrics.remove_object_key(object.as_ptr() as usize);
        }
    }

    fn element_change_state_post(
        &self,
        _ts: u64,
        element: &gst::Element,
        _change: gst::StateChange,
        result: Result<gst::StateChangeSuccess, gst::StateChangeError>,
    ) {
        if result.is_err() {
            return;
        }
        if let Some(pipeline) = element.downcast_ref::<gst::Pipeline>()
            && let Some(metrics) = self.metrics.get()
        {
            metrics.set_pipeline_state(pipeline, element.current_state());
        }
    }

    fn pad_push_pre(&self, _ts: u64, pad: &gst::Pad, buffer: &gst::Buffer) {
        self.metrics
            .record_push(pad, 1, u64::try_from(buffer.size()).unwrap_or(u64::MAX));
    }

    fn pad_push_list_pre(&self, _ts: u64, pad: &gst::Pad, list: &gst::BufferList) {
        let buffers = u64::try_from(list.len()).unwrap_or(u64::MAX);
        let bytes = list
            .iter()
            .map(|buffer| u64::try_from(buffer.size()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        self.metrics.record_push(pad, buffers, bytes);
    }
}

impl StatsdTracer {
    fn track_element(&self, element: &gst::Element) {
        let Some(metrics) = self.metrics.get() else {
            return;
        };
        if is_supported_queue(element) {
            metrics.track_queue(element);
        }
        if let Some(pipeline) = element.downcast_ref::<gst::Pipeline>() {
            metrics.track_pipeline(pipeline);
        }
    }
}

fn set_string(value: &glib::Value, target: &mut String) {
    if let Ok(value) = value.get::<String>() {
        *target = value;
    }
}

fn set_optional_string(value: &glib::Value, target: &mut Option<String>) {
    if let Ok(value) = value.get::<Option<String>>() {
        *target = value;
    }
}

fn set_u32(value: &glib::Value, target: &mut u32) {
    if let Ok(value) = value.get::<u32>() {
        *target = value;
    }
}

fn validate_settings(settings: &Settings) -> Result<WorkerConfig, String> {
    let destination = settings
        .destination
        .parse::<SocketAddr>()
        .map_err(|error| format!("Invalid numeric StatsD destination: {error}"))?;
    validate_prefix(&settings.prefix)?;
    let global_tags = parse_global_tags(settings.global_tags.as_deref())?;
    Ok(WorkerConfig {
        destination,
        prefix: settings.prefix.clone(),
        global_tags,
        flush_interval: Duration::from_millis(u64::from(settings.flush_interval_ms)),
    })
}

fn validate_prefix(prefix: &str) -> Result<(), String> {
    if prefix.is_empty() || prefix.len() > 128 {
        return Err("Invalid prefix: expected 1 through 128 ASCII bytes".to_owned());
    }
    if prefix.starts_with('.') || prefix.ends_with('.') || prefix.contains("..") {
        return Err("Invalid prefix: dots cannot be leading, trailing, or repeated".to_owned());
    }
    if !prefix
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err("Invalid prefix: unsupported character".to_owned());
    }
    Ok(())
}

fn parse_global_tags(value: Option<&str>) -> Result<Vec<(String, String)>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.len() > 512 {
        return Err("Invalid global-tags: property exceeds 512 bytes".to_owned());
    }
    if value.is_empty() {
        return Err("Invalid global-tags: empty tag list".to_owned());
    }
    let mut tags = Vec::new();
    for raw in value.split(',') {
        if tags.len() >= 16 {
            return Err("Invalid global-tags: more than 16 tags".to_owned());
        }
        let tag = raw.trim_matches(|character: char| character.is_ascii_whitespace());
        let Some((raw_key, raw_value)) = tag.split_once(':') else {
            return Err("Invalid global-tags: each tag needs one key:value delimiter".to_owned());
        };
        if raw_value.contains(':') {
            return Err("Invalid global-tags: each tag needs exactly one colon".to_owned());
        }
        let key = raw_key.trim_matches(|character: char| character.is_ascii_whitespace());
        let value = raw_value.trim_matches(|character: char| character.is_ascii_whitespace());
        if key.is_empty() || value.is_empty() {
            return Err("Invalid global-tags: keys and values must be non-empty".to_owned());
        }
        if !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            return Err("Invalid global-tags: unsupported key character".to_owned());
        }
        if value
            .chars()
            .any(|character| character == ',' || character == '|' || character.is_control())
        {
            return Err("Invalid global-tags: unsupported value character".to_owned());
        }
        if tags.iter().any(|(existing, _)| existing == key) {
            return Err(format!("Invalid global-tags: duplicate key {key}"));
        }
        tags.push((key.to_owned(), value.to_owned()));
    }
    Ok(tags)
}

fn compile_filter(name: &str, value: Option<&str>) -> Result<Option<Regex>, String> {
    value
        .map(|value| Regex::new(value).map_err(|error| format!("Invalid {name}: {error}")))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    static TRACER_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn property_defaults_and_post_construction_freeze() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::StatsdTracer>()
            .property("destination", "127.0.0.1:9")
            .property("flush-interval-ms", 100_u32)
            .property("max-pad-series", 7_u32)
            .build();
        assert_eq!(tracer.property::<String>("prefix"), DEFAULT_PREFIX);
        assert!(tracer.property::<bool>("worker-running"));
        tracer.set_property("max-pad-series", 8_u32);
        assert_eq!(tracer.property::<u32>("max-pad-series"), 7);
        tracer.imp().dispose();
        tracer.imp().dispose();
        assert!(!tracer.property::<bool>("worker-running"));
    }

    #[test]
    fn property_invalid_configuration_remains_inactive() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        for (property, value) in [
            ("destination", "localhost:8125"),
            ("prefix", ".invalid"),
            ("prefix", "invalid..prefix"),
            ("global-tags", "missing-colon"),
            ("global-tags", "a:b,a:c"),
            ("include-filter", "["),
            ("exclude-filter", "["),
        ] {
            let tracer = glib::Object::builder::<super::super::StatsdTracer>()
                .property(property, value)
                .build();
            assert!(
                !tracer.property::<bool>("worker-running"),
                "{property}={value}"
            );
        }
    }

    #[test]
    fn property_global_tags_parser_preserves_order() {
        assert_eq!(
            parse_global_tags(Some(" env:prod , region: eu ")).expect("valid tags"),
            vec![
                ("env".to_owned(), "prod".to_owned()),
                ("region".to_owned(), "eu".to_owned())
            ]
        );
        assert_eq!(parse_global_tags(Some(&"a:v,".repeat(17))).ok(), None);
        assert_eq!(parse_global_tags(Some(&"x".repeat(513))).ok(), None);
    }
}
