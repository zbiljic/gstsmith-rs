use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use gst::glib;
use gst::prelude::*;
use parking_lot::{Mutex, RwLock};
use prometheus_client::encoding::{
    EncodeLabelSet, EncodeLabelValue, LabelValueEncoder, text::encode,
};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;
use regex::Regex;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EscapedLabelValue(String);

impl From<String> for EscapedLabelValue {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl EncodeLabelValue for EscapedLabelValue {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        for character in self.0.chars() {
            match character {
                '\\' => encoder.write_str("\\\\")?,
                '"' => encoder.write_str("\\\"")?,
                '\n' => encoder.write_str("\\n")?,
                character => encoder.write_char(character)?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, EncodeLabelSet, Eq, Hash, PartialEq)]
struct PadLabels {
    element: EscapedLabelValue,
    pad: EscapedLabelValue,
}

#[derive(Clone, Debug, EncodeLabelSet, Eq, Hash, PartialEq)]
struct PipelineLabels {
    pipeline: EscapedLabelValue,
    state: EscapedLabelValue,
}

#[derive(Clone, Debug, EncodeLabelSet, Eq, Hash, PartialEq)]
struct ElementLabels {
    element: EscapedLabelValue,
}

#[derive(Clone, Debug, EncodeLabelSet, Eq, Hash, PartialEq)]
struct ReasonLabels {
    reason: EscapedLabelValue,
}

#[derive(Clone)]
struct TrackedPad {
    labels: PadLabels,
    buffers: Counter,
    bytes: Counter,
}

#[derive(Clone)]
enum PadEntry {
    Tracked(TrackedPad),
    Ignored(IgnoreReason),
}

#[derive(Clone, Copy)]
enum IgnoreReason {
    Filtered,
    SeriesLimit,
}

#[derive(Clone, Copy)]
struct QueueLimits {
    buffers: u32,
    bytes: u32,
    time: u64,
}

struct QueueEntry {
    element: glib::WeakRef<gst::Element>,
    limits: Arc<Mutex<Option<QueueLimits>>>,
    handlers: Vec<glib::SignalHandlerId>,
}

impl QueueEntry {
    fn new(element: &gst::Element) -> Self {
        let limits = Arc::new(Mutex::new(None));
        let handlers = ["max-size-buffers", "max-size-bytes", "max-size-time"]
            .into_iter()
            .map(|name| {
                let limits = Arc::clone(&limits);
                element.connect_notify(Some(name), move |_, _| *limits.lock() = None)
            })
            .collect();
        Self {
            element: element.downgrade(),
            limits,
            handlers,
        }
    }
}

impl Drop for QueueEntry {
    fn drop(&mut self) {
        if let Some(element) = self.element.upgrade() {
            for handler in self.handlers.drain(..) {
                element.disconnect(handler);
            }
        }
    }
}

struct NameWatch {
    object: glib::WeakRef<gst::Object>,
    parent: Option<usize>,
    handler: Option<glib::SignalHandlerId>,
}

impl NameWatch {
    fn new(object: &gst::Object, dirty: &Arc<AtomicBool>) -> Self {
        let dirty = Arc::clone(dirty);
        let handler = object.connect_notify(Some("name"), move |_, _| {
            dirty.store(true, Ordering::Release);
        });
        Self {
            object: object.downgrade(),
            parent: None,
            handler: Some(handler),
        }
    }
}

impl Drop for NameWatch {
    fn drop(&mut self) {
        if let Some(object) = self.object.upgrade()
            && let Some(handler) = self.handler.take()
        {
            object.disconnect(handler);
        }
    }
}

struct CachedQueue {
    key: usize,
    entry: std::sync::Weak<QueueEntry>,
    labels: ElementLabels,
}

// ponytail: one dirty flag rebuilds all queue identities after any graph edit.
// Use per-pipeline invalidation only if rebuilds across independent pipelines matter.
#[derive(Default)]
struct QueueCache {
    checked_at: Option<Instant>,
    watches: HashMap<usize, NameWatch>,
    queues: Vec<CachedQueue>,
}

impl QueueCache {
    fn watch_ancestors(
        &mut self,
        element: &gst::Element,
        dirty: &Arc<AtomicBool>,
        seen: &mut HashSet<gst::Object>,
    ) {
        let mut current = Some(element.clone().upcast::<gst::Object>());
        while let Some(object) = current {
            if !seen.insert(object.clone()) {
                break;
            }
            let key = object.as_ptr() as usize;
            if self
                .watches
                .get(&key)
                .is_some_and(|watch| watch.object.upgrade().is_none())
            {
                self.watches.remove(&key);
            }
            current = object.parent();
            let watch = self
                .watches
                .entry(key)
                .or_insert_with(|| NameWatch::new(&object, dirty));
            watch.parent = current.as_ref().map(|parent| parent.as_ptr() as usize);
        }
    }
}

struct PipelineEntry {
    pipeline: glib::WeakRef<gst::Pipeline>,
    labels: Vec<PipelineLabels>,
    gauges: Vec<Gauge>,
    observed_transition: std::sync::atomic::AtomicBool,
}

pub(crate) struct Metrics {
    registry: Mutex<Registry>,
    pad_buffers: Family<PadLabels, Counter>,
    pad_bytes: Family<PadLabels, Counter>,
    pipeline_state: Family<PipelineLabels, Gauge>,
    queue_level_buffers: Family<ElementLabels, Gauge>,
    queue_level_bytes: Family<ElementLabels, Gauge>,
    queue_level_seconds: Family<ElementLabels, Gauge<f64, std::sync::atomic::AtomicU64>>,
    queue_capacity_buffers: Family<ElementLabels, Gauge>,
    queue_capacity_bytes: Family<ElementLabels, Gauge>,
    queue_capacity_seconds: Family<ElementLabels, Gauge<f64, std::sync::atomic::AtomicU64>>,
    untracked_series_limit: Counter,
    encoding_failures: Counter,
    // Push events only pin and read this map. Rare writes are serialized separately so the
    // cardinality limit remains exact without putting a shared lock on the streaming hot path.
    pads: papaya::HashMap<usize, PadEntry>,
    tracked_pad_count: Mutex<usize>,
    pipelines: RwLock<HashMap<usize, PipelineEntry>>,
    queues: RwLock<HashMap<usize, Arc<QueueEntry>>>,
    queue_cache: Mutex<QueueCache>,
    queue_identities_dirty: Arc<AtomicBool>,
    queue_labels: Mutex<HashSet<ElementLabels>>,
    include_filter: Option<Regex>,
    exclude_filter: Option<Regex>,
    max_pad_series: usize,
    #[cfg(test)]
    fail_next_encoding: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
pub(crate) struct MetricsSlot(OnceLock<Arc<Metrics>>);

impl MetricsSlot {
    pub(crate) fn install(&self, metrics: Arc<Metrics>) -> bool {
        self.0.set(metrics).is_ok()
    }

    pub(crate) fn get(&self) -> Option<&Metrics> {
        self.0.get().map(Arc::as_ref)
    }

    pub(crate) fn record_push(&self, pad: &gst::Pad, buffers: u64, bytes: u64) {
        if let Some(metrics) = self.get() {
            metrics.update_pad(pad, buffers, bytes);
        }
    }
}

impl Metrics {
    pub(crate) fn new(
        include_filter: Option<Regex>,
        exclude_filter: Option<Regex>,
        max_pad_series: usize,
    ) -> Arc<Self> {
        let pad_buffers = Family::default();
        let pad_bytes = Family::default();
        let pipeline_state = Family::default();
        let queue_level_buffers = Family::default();
        let queue_level_bytes = Family::default();
        let queue_level_seconds = Family::default();
        let queue_capacity_buffers = Family::default();
        let queue_capacity_bytes = Family::default();
        let queue_capacity_seconds = Family::default();
        let untracked: Family<ReasonLabels, Counter> = Family::default();
        let untracked_series_limit = untracked
            .get_or_create(&ReasonLabels {
                reason: "series_limit".to_owned().into(),
            })
            .clone();
        let encoding_failures = Counter::default();
        let mut registry = Registry::default();

        registry.register(
            "gstsmith_gstreamer_pad_push_buffers",
            "Buffers attempted through pad-push-pre",
            pad_buffers.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_pad_push_bytes",
            "Bytes attempted through pad-push-pre",
            pad_bytes.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_pipeline_state",
            "One-hot current GStreamer pipeline state",
            pipeline_state.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_level_buffers",
            "Current number of queued buffers",
            queue_level_buffers.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_level_bytes",
            "Current number of queued bytes",
            queue_level_bytes.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_level_seconds",
            "Current queued time in seconds",
            queue_level_seconds.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_capacity_buffers",
            "Configured queue capacity in buffers",
            queue_capacity_buffers.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_capacity_bytes",
            "Configured queue capacity in bytes",
            queue_capacity_bytes.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_queue_capacity_seconds",
            "Configured queue capacity in seconds",
            queue_capacity_seconds.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_untracked_pad_events",
            "Push events not represented because the pad series limit was reached",
            untracked.clone(),
        );
        registry.register(
            "gstsmith_gstreamer_scrape_encoding_failures",
            "Registry encoding failures observed before the current scrape",
            encoding_failures.clone(),
        );

        Arc::new(Self {
            registry: Mutex::new(registry),
            pad_buffers,
            pad_bytes,
            pipeline_state,
            queue_level_buffers,
            queue_level_bytes,
            queue_level_seconds,
            queue_capacity_buffers,
            queue_capacity_bytes,
            queue_capacity_seconds,
            untracked_series_limit,
            encoding_failures,
            pads: papaya::HashMap::new(),
            tracked_pad_count: Mutex::new(0),
            pipelines: RwLock::default(),
            queues: RwLock::default(),
            queue_cache: Mutex::default(),
            queue_identities_dirty: Arc::default(),
            queue_labels: Mutex::default(),
            include_filter,
            exclude_filter,
            max_pad_series,
            #[cfg(test)]
            fail_next_encoding: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn included(&self, identity: &str) -> bool {
        self.include_filter
            .as_ref()
            .is_none_or(|filter| filter.is_match(identity))
            && self
                .exclude_filter
                .as_ref()
                .is_none_or(|filter| !filter.is_match(identity))
    }

    pub(crate) fn update_pad(&self, pad: &gst::Pad, buffers: u64, bytes: u64) {
        let key = pad.as_ptr() as usize;
        let pads = self.pads.pin();
        if let Some(entry) = pads.get(&key) {
            Self::increment_entry(entry, buffers, bytes, &self.untracked_series_limit);
            return;
        }
        drop(pads);

        let Some(element) = pad
            .parent()
            .and_then(|parent| parent.downcast::<gst::Element>().ok())
        else {
            return;
        };
        let element_path = element.path_string().to_string();
        let pad_name = pad.name().to_string();
        let identity = format!("{element_path}:{pad_name}");
        let labels = PadLabels {
            element: element_path.into(),
            pad: pad_name.into(),
        };
        let included = self.included(&identity);
        let mut tracked_pad_count = self.tracked_pad_count.lock();
        let pads = self.pads.pin();
        if let Some(entry) = pads.get(&key) {
            Self::increment_entry(entry, buffers, bytes, &self.untracked_series_limit);
            return;
        }
        let entry = if !included || *tracked_pad_count >= self.max_pad_series {
            if included {
                PadEntry::Ignored(IgnoreReason::SeriesLimit)
            } else {
                PadEntry::Ignored(IgnoreReason::Filtered)
            }
        } else {
            *tracked_pad_count += 1;
            PadEntry::Tracked(TrackedPad {
                buffers: self.pad_buffers.get_or_create(&labels).clone(),
                bytes: self.pad_bytes.get_or_create(&labels).clone(),
                labels,
            })
        };
        let entry = pads.get_or_insert(key, entry);
        Self::increment_entry(entry, buffers, bytes, &self.untracked_series_limit);
    }

    fn increment_entry(
        entry: &PadEntry,
        buffers: u64,
        bytes: u64,
        untracked_series_limit: &Counter,
    ) {
        match entry {
            PadEntry::Tracked(entry) => {
                entry.buffers.inc_by(buffers);
                entry.bytes.inc_by(bytes);
            }
            PadEntry::Ignored(IgnoreReason::SeriesLimit) => {
                untracked_series_limit.inc_by(buffers);
            }
            PadEntry::Ignored(IgnoreReason::Filtered) => {}
        }
    }

    pub(crate) fn remove_pad(&self, pad: &gst::Pad) {
        self.remove_pad_key(pad.as_ptr() as usize);
    }

    pub(crate) fn remove_object_key(&self, key: usize) {
        self.invalidate_queue_identities();
        self.remove_pad_key(key);
        self.remove_pipeline_key(key);
        self.remove_queue_key(key);
    }

    fn remove_pad_key(&self, key: usize) {
        let mut tracked_pad_count = self.tracked_pad_count.lock();
        let pads = self.pads.pin();
        if let Some(PadEntry::Tracked(entry)) = pads.remove(&key) {
            *tracked_pad_count = tracked_pad_count.saturating_sub(1);
            self.pad_buffers.remove(&entry.labels);
            self.pad_bytes.remove(&entry.labels);
        }
    }

    pub(crate) fn track_pipeline(&self, pipeline: &gst::Pipeline) {
        let identity = pipeline.name().to_string();
        if !self.included(&identity) {
            return;
        }
        let key = pipeline.as_ptr() as usize;
        self.pipelines.write().entry(key).or_insert_with(|| {
            let mut labels = Vec::with_capacity(4);
            let mut gauges = Vec::with_capacity(4);
            for state in ["null", "ready", "paused", "playing"] {
                let label = PipelineLabels {
                    pipeline: identity.clone().into(),
                    state: state.to_owned().into(),
                };
                let gauge = self.pipeline_state.get_or_create(&label).clone();
                gauge.set(0);
                labels.push(label);
                gauges.push(gauge);
            }
            PipelineEntry {
                pipeline: pipeline.downgrade(),
                labels,
                gauges,
                observed_transition: std::sync::atomic::AtomicBool::new(false),
            }
        });
    }

    pub(crate) fn set_pipeline_state(&self, pipeline: &gst::Pipeline, state: gst::State) {
        self.track_pipeline(pipeline);
        let key = pipeline.as_ptr() as usize;
        if let Some(entry) = self.pipelines.read().get(&key) {
            entry
                .observed_transition
                .store(true, std::sync::atomic::Ordering::Release);
            Self::set_pipeline_entry_state(entry, state);
        }
    }

    fn set_pipeline_entry_state(entry: &PipelineEntry, state: gst::State) {
        let active = match state {
            gst::State::Null => 0,
            gst::State::Ready => 1,
            gst::State::Paused => 2,
            gst::State::Playing => 3,
            gst::State::VoidPending => return,
        };
        for (index, gauge) in entry.gauges.iter().enumerate() {
            gauge.set(i64::from(index == active));
        }
    }

    fn refresh_pipeline_states(&self) {
        let snapshot = self
            .pipelines
            .read()
            .iter()
            .filter(|(_key, entry)| {
                entry
                    .observed_transition
                    .load(std::sync::atomic::Ordering::Acquire)
            })
            .map(|(key, entry)| (*key, entry.pipeline.clone()))
            .collect::<Vec<_>>();
        let mut stale = Vec::new();
        for (key, weak) in snapshot {
            let Some(pipeline) = weak.upgrade() else {
                stale.push(key);
                continue;
            };
            let current = pipeline.current_state();
            if let Some(entry) = self.pipelines.read().get(&key) {
                Self::set_pipeline_entry_state(entry, current);
            }
        }
        for key in stale {
            self.remove_pipeline_key(key);
        }
    }

    fn remove_pipeline_key(&self, key: usize) {
        if let Some(entry) = self.pipelines.write().remove(&key) {
            for label in entry.labels {
                self.pipeline_state.remove(&label);
            }
        }
    }

    pub(crate) fn track_queue(&self, element: &gst::Element) {
        self.queues
            .write()
            .entry(element.as_ptr() as usize)
            .or_insert_with(|| Arc::new(QueueEntry::new(element)));
        self.invalidate_queue_identities();
    }

    pub(crate) fn invalidate_queue_identities(&self) {
        self.queue_identities_dirty.store(true, Ordering::Release);
    }

    fn refresh_queue_cache(&self, cache: &mut QueueCache) {
        // Clear before rebuilding so notifications during the rebuild remain pending.
        // Direct GstObject parenting has no notification on GStreamer 1.24. Recheck
        // at the first sample at least 30 seconds after the previous check.
        let dirty = self.queue_identities_dirty.swap(false, Ordering::AcqRel);
        if !dirty
            && cache
                .checked_at
                .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(30))
        {
            return;
        }
        cache.checked_at = Some(Instant::now());
        if !dirty
            && cache.watches.values().all(|watch| {
                watch.object.upgrade().is_some_and(|object| {
                    object.parent().map(|parent| parent.as_ptr() as usize) == watch.parent
                })
            })
        {
            return;
        }
        let snapshot = self
            .queues
            .read()
            .iter()
            .map(|(key, entry)| (*key, Arc::clone(entry)))
            .collect::<Vec<_>>();
        cache.queues.clear();
        let mut seen = HashSet::new();
        let mut parent_paths = HashMap::new();
        for (key, entry) in snapshot {
            let Some(element) = entry.element.upgrade() else {
                self.remove_queue_key(key);
                continue;
            };
            // Watch excluded queues too: a rename or ancestor move can include them.
            cache.watch_ancestors(&element, &self.queue_identities_dirty, &mut seen);
            let identity = if let Some(parent) = element.parent() {
                let prefix = parent_paths
                    .entry(parent.clone())
                    .or_insert_with(|| parent.path_string());
                // The tracked core queue/queue2 types both use "/" path separators.
                format!("{prefix}/{}:{}", element.type_().name(), element.name())
            } else {
                element.path_string().to_string()
            };
            if self.included(&identity) {
                cache.queues.push(CachedQueue {
                    key,
                    entry: Arc::downgrade(&entry),
                    labels: ElementLabels {
                        element: identity.into(),
                    },
                });
            }
        }
        cache.watches.retain(|_, watch| {
            watch
                .object
                .upgrade()
                .is_some_and(|object| seen.contains(&object))
        });
    }

    fn refresh_queues(&self) {
        // Reuse live series and remove obsolete labels after sampling current paths.
        let mut previous_labels = self.queue_labels.lock();
        let mut cache = self.queue_cache.lock();
        self.refresh_queue_cache(&mut cache);
        let mut current_labels = HashSet::with_capacity(cache.queues.len());
        let mut stale = Vec::new();
        for cached in &cache.queues {
            let Some(entry) = cached.entry.upgrade() else {
                continue;
            };
            let Some(element) = entry.element.upgrade() else {
                stale.push(cached.key);
                continue;
            };
            // Serialize refresh with notifications so a concurrent setter cannot lose invalidation.
            let limits = *entry.limits.lock().get_or_insert_with(|| QueueLimits {
                buffers: element.property("max-size-buffers"),
                bytes: element.property("max-size-bytes"),
                time: element.property("max-size-time"),
            });
            let labels = &cached.labels;
            let values = (
                element.property::<u32>("current-level-buffers"),
                element.property::<u32>("current-level-bytes"),
                element.property::<u64>("current-level-time"),
                limits.buffers,
                limits.bytes,
                limits.time,
            );
            self.queue_level_buffers
                .get_or_create(labels)
                .set(i64::from(values.0));
            self.queue_level_bytes
                .get_or_create(labels)
                .set(i64::from(values.1));
            self.queue_level_seconds
                .get_or_create(labels)
                .set(std::time::Duration::from_nanos(values.2).as_secs_f64());
            self.queue_capacity_buffers
                .get_or_create(labels)
                .set(i64::from(values.3));
            self.queue_capacity_bytes
                .get_or_create(labels)
                .set(i64::from(values.4));
            self.queue_capacity_seconds
                .get_or_create(labels)
                .set(std::time::Duration::from_nanos(values.5).as_secs_f64());
            current_labels.insert(labels.clone());
        }
        for labels in previous_labels.difference(&current_labels) {
            self.queue_level_buffers.remove(labels);
            self.queue_level_bytes.remove(labels);
            self.queue_level_seconds.remove(labels);
            self.queue_capacity_buffers.remove(labels);
            self.queue_capacity_bytes.remove(labels);
            self.queue_capacity_seconds.remove(labels);
        }
        *previous_labels = current_labels;
        for key in stale {
            self.remove_queue_key(key);
        }
    }

    fn remove_queue_key(&self, key: usize) {
        // Disconnect handlers and release GStreamer references after unlocking the map.
        let entry = self.queues.write().remove(&key);
        if entry.is_some() {
            self.invalidate_queue_identities();
        }
        drop(entry);
    }

    pub(crate) fn encode(&self) -> Result<String, std::fmt::Error> {
        // Serialize refresh and encoding so concurrent scrapes see complete queue gauges.
        let registry = self.registry.lock();
        self.refresh_pipeline_states();
        self.refresh_queues();
        #[cfg(test)]
        if self
            .fail_next_encoding
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.encoding_failures.inc();
            return Err(std::fmt::Error);
        }
        let mut output = String::new();
        if let Err(error) = encode(&mut output, &registry) {
            self.encoding_failures.inc();
            return Err(error);
        }
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn fail_next_encoding_for_test(&self) {
        self.fail_next_encoding
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_encode_exact_names_and_eof() {
        let metrics = Metrics::new(None, None, 256);
        gst::init().expect("initializing GStreamer");
        let element = gst::ElementFactory::make("identity")
            .build()
            .expect("constructing identity");
        let pad = element.static_pad("src").expect("identity source pad");
        metrics.update_pad(&pad, 1, 10);
        let pipeline = gst::Pipeline::builder().name("family-pipeline").build();
        metrics.track_pipeline(&pipeline);
        let queue = gst::ElementFactory::make("queue")
            .name("family-queue")
            .build()
            .expect("constructing family queue");
        metrics.track_queue(&queue);
        metrics
            .pad_buffers
            .get_or_create(&PadLabels {
                element: "quoted\"slash\\element".to_owned().into(),
                pad: "line\npad".to_owned().into(),
            })
            .inc();
        let output = metrics.encode().expect("String encoding cannot fail");
        for expected in [
            "# HELP gstsmith_gstreamer_pad_push_buffers Buffers attempted through pad-push-pre.",
            "# TYPE gstsmith_gstreamer_pad_push_buffers counter",
            "# HELP gstsmith_gstreamer_pad_push_bytes Bytes attempted through pad-push-pre.",
            "# TYPE gstsmith_gstreamer_pad_push_bytes counter",
            "# HELP gstsmith_gstreamer_pipeline_state One-hot current GStreamer pipeline state.",
            "# TYPE gstsmith_gstreamer_pipeline_state gauge",
            "# HELP gstsmith_gstreamer_queue_level_buffers Current number of queued buffers.",
            "# TYPE gstsmith_gstreamer_queue_level_buffers gauge",
            "# HELP gstsmith_gstreamer_queue_level_bytes Current number of queued bytes.",
            "# TYPE gstsmith_gstreamer_queue_level_bytes gauge",
            "# HELP gstsmith_gstreamer_queue_level_seconds Current queued time in seconds.",
            "# TYPE gstsmith_gstreamer_queue_level_seconds gauge",
            "# HELP gstsmith_gstreamer_queue_capacity_buffers Configured queue capacity in buffers.",
            "# TYPE gstsmith_gstreamer_queue_capacity_buffers gauge",
            "# HELP gstsmith_gstreamer_queue_capacity_bytes Configured queue capacity in bytes.",
            "# TYPE gstsmith_gstreamer_queue_capacity_bytes gauge",
            "# HELP gstsmith_gstreamer_queue_capacity_seconds Configured queue capacity in seconds.",
            "# TYPE gstsmith_gstreamer_queue_capacity_seconds gauge",
            "# HELP gstsmith_gstreamer_untracked_pad_events Push events not represented because the pad series limit was reached.",
            "# TYPE gstsmith_gstreamer_untracked_pad_events counter",
            "# HELP gstsmith_gstreamer_scrape_encoding_failures Registry encoding failures observed before the current scrape.",
            "# TYPE gstsmith_gstreamer_scrape_encoding_failures counter",
        ] {
            assert!(
                output.lines().any(|line| line == expected),
                "missing {expected:?} in:\n{output}"
            );
        }
        assert!(output.lines().any(|line| {
            line == "gstsmith_gstreamer_pad_push_buffers_total{element=\"quoted\\\"slash\\\\element\",pad=\"line\\npad\"} 1"
        }), "{output}");
        assert!(output.ends_with("# EOF\n"));
    }

    #[test]
    fn concurrent_counter_increments_are_exact() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let element = gst::ElementFactory::make("identity")
            .build()
            .expect("constructing identity");
        let pad = element.static_pad("src").expect("identity source pad");
        let threads = (0..4)
            .map(|_| {
                let metrics = Arc::clone(&metrics);
                let pad = pad.clone();
                std::thread::spawn(move || {
                    for _ in 0..1_000 {
                        metrics.update_pad(&pad, 1, 10);
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("counter worker");
        }
        let output = metrics.encode().expect("encoding metrics");
        let element_path = element.path_string();
        assert!(output.lines().any(|line| {
            line == format!(
                "gstsmith_gstreamer_pad_push_buffers_total{{element=\"{element_path}\",pad=\"src\"}} 4000"
            )
        }));
        assert!(output.lines().any(|line| {
            line == format!(
                "gstsmith_gstreamer_pad_push_bytes_total{{element=\"{element_path}\",pad=\"src\"}} 40000"
            )
        }));
    }

    #[test]
    fn concurrent_pad_registration_preserves_the_series_limit() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 3);
        let elements = (0..8)
            .map(|index| {
                gst::ElementFactory::make("identity")
                    .name(format!("concurrent-{index}"))
                    .build()
                    .expect("constructing identity")
            })
            .collect::<Vec<_>>();
        let barrier = Arc::new(std::sync::Barrier::new(elements.len()));
        let threads = elements
            .iter()
            .map(|element| {
                let metrics = Arc::clone(&metrics);
                let barrier = Arc::clone(&barrier);
                let pad = element.static_pad("src").expect("identity source pad");
                std::thread::spawn(move || {
                    barrier.wait();
                    metrics.update_pad(&pad, 1, 10);
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("registration worker");
        }

        let output = metrics.encode().expect("encoding metrics");
        assert_eq!(
            output
                .lines()
                .filter(|line| { line.starts_with("gstsmith_gstreamer_pad_push_buffers_total{") })
                .count(),
            3
        );
        assert!(output.contains("reason=\"series_limit\"} 5"));
    }

    #[test]
    fn pad_series_limit_filtering_and_removal_are_stable() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let first = gst::ElementFactory::make("identity")
            .name("first")
            .build()
            .expect("constructing first identity");
        let second = gst::ElementFactory::make("identity")
            .name("second")
            .build()
            .expect("constructing second identity");
        let third = gst::ElementFactory::make("identity")
            .name("third")
            .build()
            .expect("constructing third identity");
        let first_pad = first.static_pad("src").expect("first source pad");
        let second_pad = second.static_pad("src").expect("second source pad");
        let third_pad = third.static_pad("src").expect("third source pad");

        metrics.update_pad(&first_pad, 1, 10);
        metrics.update_pad(&second_pad, 2, 20);
        metrics.remove_pad(&first_pad);
        metrics.update_pad(&second_pad, 3, 30);
        metrics.update_pad(&third_pad, 1, 10);

        let output = metrics.encode().expect("encoding capped metrics");
        assert!(!output.contains("first"));
        assert!(!output.contains("second"));
        assert!(output.contains("third"));
        assert!(output.contains("reason=\"series_limit\"} 5"));

        let filtered = Metrics::new(Some(Regex::new("allowed").expect("valid regex")), None, 1);
        filtered.update_pad(&second_pad, 7, 70);
        let output = filtered.encode().expect("encoding filtered metrics");
        assert!(!output.contains("second"));
        assert!(output.contains("reason=\"series_limit\"} 0"));
    }

    #[test]
    fn pipeline_states_start_zero_then_become_one_hot_and_are_removed() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let pipeline = gst::Pipeline::builder().name("state-pipeline").build();
        metrics.track_pipeline(&pipeline);
        let initial = metrics.encode().expect("encoding initial pipeline state");
        for state in ["null", "ready", "paused", "playing"] {
            assert!(initial.contains(&format!(
                "pipeline=\"state-pipeline\",state=\"{state}\"}} 0"
            )));
        }

        pipeline
            .set_state(gst::State::Playing)
            .expect("setting actual pipeline state");
        let (result, current, pending) = pipeline.state(gst::ClockTime::from_seconds(2));
        result.expect("waiting for actual pipeline state");
        assert_eq!(current, gst::State::Playing);
        assert_eq!(pending, gst::State::VoidPending);
        metrics.set_pipeline_state(&pipeline, current);
        let playing = metrics.encode().expect("encoding playing pipeline state");
        assert!(playing.contains("pipeline=\"state-pipeline\",state=\"playing\"} 1"));
        assert!(playing.contains("pipeline=\"state-pipeline\",state=\"paused\"} 0"));
        pipeline
            .set_state(gst::State::Null)
            .expect("stopping pipeline");
        metrics.remove_object_key(pipeline.as_ptr() as usize);
        assert!(
            !metrics
                .encode()
                .expect("encoding removed pipeline")
                .contains("state-pipeline")
        );
    }

    #[test]
    fn queue_capacity_uses_base_units_and_escaped_labels() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let queue = gst::ElementFactory::make("queue")
            .name("quoted\"queue")
            .property("max-size-buffers", 23_u32)
            .property("max-size-bytes", 42_u32)
            .property("max-size-time", 2_500_000_000_u64)
            .build()
            .expect("constructing queue");
        metrics.track_queue(&queue);
        let output = metrics.encode().expect("encoding queue metrics");
        assert!(output.contains("gstsmith_gstreamer_queue_capacity_buffers"));
        assert!(output.contains(" 23"));
        assert!(output.contains("gstsmith_gstreamer_queue_capacity_bytes"));
        assert!(output.contains(" 42"));
        assert!(output.contains("gstsmith_gstreamer_queue_capacity_seconds"));
        assert!(output.contains(" 2.5"));
    }

    #[test]
    fn queue_metrics_follow_nested_bin_paths_and_filters() {
        gst::init().expect("initializing GStreamer");
        let include = Regex::new("GstPipeline:included").expect("include regex");
        let exclude = Regex::new("excluded").expect("exclude regex");
        for factory in ["queue", "queue2"] {
            for filtered in [false, true] {
                let metrics = Metrics::new(
                    filtered.then(|| include.clone()),
                    filtered.then(|| exclude.clone()),
                    1,
                );
                let sample = |queue: &gst::Element, capacity: u32| {
                    format!(
                        "gstsmith_gstreamer_queue_capacity_buffers{{element=\"{}\"}} {capacity}",
                        queue.path_string()
                    )
                };
                let [(first_bin, first), (second_bin, second)] = [11_u32, 22].map(|capacity| {
                    let bin = gst::Bin::builder().name("branch").build();
                    let queue = gst::ElementFactory::make(factory)
                        .name("observed")
                        .property("max-size-buffers", capacity)
                        .build()
                        .expect("constructing queue");
                    bin.add(&queue)
                        .expect("adding queue before parenting its bin");
                    metrics.track_queue(&queue);
                    (bin, queue)
                });
                let initial_label = format!("element=\"{}\"", first.path_string());
                let _initial = metrics.encode().expect("sampling unparented bins");
                let first_pipeline = gst::Pipeline::builder().name("included_first").build();
                let second_pipeline = gst::Pipeline::builder().name("included_second").build();
                let excluded = gst::Pipeline::builder().name("included_excluded").build();
                first_pipeline.add(&first_bin).expect("parenting first bin");
                second_pipeline
                    .add(&second_bin)
                    .expect("parenting second bin");

                metrics.invalidate_queue_identities();
                let output = metrics.encode().expect("sampling parented bins");
                assert!(output.contains(&sample(&first, 11)), "{output}");
                assert!(output.contains(&sample(&second, 22)), "{output}");
                assert!(!output.contains(&initial_label), "{output}");

                let old_sample = sample(&first, 11);
                first_pipeline
                    .remove(&first_bin)
                    .expect("unparenting first bin");
                excluded
                    .add(&first_bin)
                    .expect("moving bin into excluded pipeline");
                metrics.invalidate_queue_identities();
                let output = metrics.encode().expect("sampling reparented bin");
                assert!(!output.contains(&old_sample), "{output}");
                assert_eq!(output.contains(&sample(&first, 11)), !filtered, "{output}");
                assert!(output.contains(&sample(&second, 22)), "{output}");

                excluded
                    .remove(&first_bin)
                    .expect("unparenting excluded bin");
                first_pipeline
                    .add(&first_bin)
                    .expect("restoring included bin");
                let second_sample = sample(&second, 22);
                second_bin.remove(&second).expect("removing second queue");
                metrics.remove_object_key(second.as_ptr() as usize);
                let output = metrics.encode().expect("sampling after queue removal");
                assert!(output.contains(&sample(&first, 11)), "{output}");
                assert!(!output.contains(&second_sample), "{output}");
            }
        }
    }

    #[test]
    fn queue_cache_handles_notifications_verification_and_cleanup() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let root = gst::Pipeline::builder().name("root").build();
        let outer = gst::Bin::builder().name("outer").build();
        let queue = gst::ElementFactory::make("queue")
            .name("observed")
            .property("max-size-buffers", 11_u32)
            .build()
            .expect("constructing queue");
        root.add(&outer).expect("parenting outer bin");
        outer.add(&queue).expect("parenting queue");
        metrics.track_queue(&queue);
        let sample = || {
            let output = metrics.encode().expect("sampling queue");
            assert!(
                output.contains(&format!(
                    "gstsmith_gstreamer_queue_capacity_buffers{{element=\"{}\"}} 11",
                    queue.path_string()
                )),
                "{output}"
            );
        };
        sample();
        let checked_at = metrics.queue_cache.lock().checked_at;
        sample();
        assert_eq!(
            metrics.queue_cache.lock().checked_at,
            checked_at,
            "stable samples reuse identities"
        );

        // Parent links return to the same values, but the name watch must invalidate.
        root.remove(&outer).expect("detaching ancestor");
        outer.set_property("name", "renamed:outer/branch");
        root.add(&outer).expect("restoring ancestor");
        assert!(metrics.queue_identities_dirty.load(Ordering::Acquire));
        sample();

        // No native bin hook is emitted here. Verify the documented fallback delay
        // without sleeping: advancing the deadline must discover the missed change.
        let before = metrics.encode().expect("sampling queue");
        let owner = gst::Bin::builder().name("owner").build();
        root.set_property("parent", &owner);
        assert_eq!(metrics.encode().expect("sampling queue"), before);
        metrics.queue_cache.lock().checked_at = None;
        sample();
        root.unparent();
        metrics.queue_cache.lock().checked_at = None;
        sample();

        std::thread::scope(|scope| {
            scope.spawn(|| {
                for index in 0..32 {
                    root.set_property("name", format!("root_{index}"));
                }
            });
            for _sample in 0..32 {
                let _output = metrics.encode().expect("sampling queue");
            }
        });
        sample();
        let weak_root = root.downgrade();
        drop(root);
        assert!(
            weak_root.upgrade().is_none(),
            "cache must not own ancestors"
        );
        metrics.invalidate_queue_identities(); // ObjectDestroyed hook in a live tracer.
        sample();

        metrics.remove_object_key(queue.as_ptr() as usize);
        let _output = metrics.encode().expect("sampling queue");
        assert!(metrics.queue_cache.lock().watches.is_empty());
        assert_eq!(
            Arc::strong_count(&metrics.queue_identities_dirty),
            1,
            "pruning must disconnect name handlers on still-live objects"
        );
        metrics.track_queue(&queue);
        sample();
        let dirty = Arc::downgrade(&metrics.queue_identities_dirty);
        drop(metrics);
        assert!(
            dirty.upgrade().is_none(),
            "collector destruction must disconnect name handlers"
        );
    }

    #[test]
    fn queue_limits_cache_handles_updates_and_tracking_lifecycle() {
        gst::init().expect("initializing GStreamer");
        for factory in ["queue", "queue2"] {
            let metrics = Metrics::new(None, None, 1);
            let queue = gst::ElementFactory::make(factory)
                .build()
                .expect("constructing queue");
            let key = queue.as_ptr() as usize;
            metrics.track_queue(&queue);
            let cache = Arc::downgrade(
                &metrics
                    .queues
                    .read()
                    .get(&key)
                    .expect("tracked queue")
                    .limits,
            );
            let set_limits = |value: u32| {
                queue.set_property("max-size-buffers", value);
                queue.set_property("max-size-bytes", value * 1024);
                queue.set_property("max-size-time", u64::from(value) * 1_000_000_000);
            };
            let assert_limits = |value: u32| {
                let output = metrics.encode().expect("sampling changed limits");
                for (suffix, expected) in [
                    ("buffers", u64::from(value)),
                    ("bytes", u64::from(value) * 1024),
                    ("seconds", u64::from(value)),
                ] {
                    assert!(output.contains(&format!(
                            "gstsmith_gstreamer_queue_capacity_{suffix}{{element=\"{}\"}} {expected}",
                            queue.path_string()
                        )), "{output}");
                }
            };
            for value in [0, 1] {
                set_limits(value);
                assert_limits(value);
            }
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    for value in 2..=32 {
                        set_limits(value);
                    }
                });
                for _sample in 0..32 {
                    let _snapshot = metrics.encode().expect("sampling queue");
                }
            });
            assert_limits(32);

            metrics.remove_object_key(key);
            assert!(
                cache.upgrade().is_none(),
                "removal must disconnect notification handlers"
            );
            set_limits(33);
            metrics.track_queue(&queue);
            assert_limits(33);
            let cache = Arc::downgrade(
                &metrics
                    .queues
                    .read()
                    .get(&key)
                    .expect("retracked queue")
                    .limits,
            );
            drop(metrics);
            assert!(
                cache.upgrade().is_none(),
                "collector destruction must disconnect handlers"
            );
        }
    }

    #[test]
    fn queue_sampling_refreshes_sibling_paths_and_values() {
        gst::init().expect("initializing GStreamer");
        for factory in ["queue", "queue2"] {
            let metrics = Metrics::new(None, None, 1);
            let pipeline = gst::Pipeline::new();
            let bin = gst::Bin::builder().name("nested:bin/branch").build();
            pipeline.add(&bin).expect("parenting bin");
            let queues =
                [("first:queue/name", 11_u32), ("second\"queue", 22)].map(|(name, capacity)| {
                    let queue = gst::ElementFactory::make(factory)
                        .name(name)
                        .build()
                        .expect("constructing queue");
                    bin.add(&queue).expect("parenting queue");
                    metrics.track_queue(&queue);
                    (queue, capacity)
                });
            for (name, offset) in [("original", 0), ("renamed\"/pipeline", 1)] {
                pipeline.set_property("name", name);
                for (queue, capacity) in &queues {
                    queue.set_property("max-size-buffers", capacity + offset);
                }
                let output = metrics.encode().expect("sampling sibling queues");
                assert_eq!(
                    output
                        .lines()
                        .filter(
                            |line| line.starts_with("gstsmith_gstreamer_queue_capacity_buffers{")
                        )
                        .count(),
                    queues.len()
                );
                for (queue, capacity) in &queues {
                    let labels = ElementLabels {
                        element: queue.path_string().to_string().into(),
                    };
                    let gauge = metrics
                        .queue_capacity_buffers
                        .get(&labels)
                        .expect("labels match the native GStreamer path");
                    assert_eq!(gauge.get(), i64::from(capacity + offset));
                }
            }
        }
    }

    #[test]
    fn queue_tracker_drops_stale_weak_entries() {
        gst::init().expect("initializing GStreamer");
        let metrics = Metrics::new(None, None, 1);
        let queue = gst::ElementFactory::make("queue")
            .name("temporary-queue")
            .build()
            .expect("constructing queue");
        let identity = queue.path_string().to_string();
        metrics.track_queue(&queue);
        assert!(
            metrics
                .encode()
                .expect("encoding live queue")
                .contains(&identity)
        );

        drop(queue);
        assert!(
            !metrics
                .encode()
                .expect("encoding after queue destruction")
                .contains(&identity)
        );
    }
}
