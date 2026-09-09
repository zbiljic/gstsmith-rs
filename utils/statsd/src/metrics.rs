use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use std::time::Instant;

use gst::glib;
use gst::prelude::*;
use parking_lot::{Mutex, RwLock};
use regex::Regex;

pub(crate) const MAX_TAG_VALUE_BYTES: usize = 192;

#[derive(Debug)]
pub(crate) struct PadStats {
    pub(crate) id: u64,
    pub(crate) element: String,
    pub(crate) pad: String,
    pub(crate) buffers: AtomicU64,
    pub(crate) bytes: AtomicU64,
}

enum PadEntry {
    Tracked(Arc<PadStats>),
    Ignored(IgnoreReason),
}

#[derive(Clone, Copy)]
enum IgnoreReason {
    Filtered,
    SeriesLimit,
    SeriesIdOverflow,
}

struct RegistrationState {
    active: usize,
    next_id: u64,
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
    labels: String,
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
    name: String,
    observed_transition: AtomicBool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PipelineSnapshot {
    pub(crate) pipeline: String,
    pub(crate) state: gst::State,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QueueSnapshot {
    pub(crate) element: String,
    pub(crate) level_buffers: u32,
    pub(crate) level_bytes: u32,
    pub(crate) level_seconds: f64,
    pub(crate) capacity_buffers: u32,
    pub(crate) capacity_bytes: u32,
    pub(crate) capacity_seconds: f64,
}

pub(crate) struct Metrics {
    // Push events pin and read this map. Cold writes serialize through registration so the active
    // cap and series identifiers stay exact without a streaming-path shared lock.
    pads: papaya::HashMap<usize, PadEntry>,
    registration: Mutex<RegistrationState>,
    pipelines: RwLock<HashMap<usize, PipelineEntry>>,
    queues: RwLock<HashMap<usize, Arc<QueueEntry>>>,
    queue_cache: Mutex<QueueCache>,
    queue_identities_dirty: Arc<AtomicBool>,
    include_filter: Option<Regex>,
    exclude_filter: Option<Regex>,
    max_pad_series: usize,
    retirement: SyncSender<Arc<PadStats>>,
    pub(crate) untracked_series_limit: AtomicU64,
    pub(crate) export_emit_errors: AtomicU64,
    pub(crate) export_flush_errors: AtomicU64,
    pub(crate) dropped_retirements: AtomicU64,
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
    ) -> (Arc<Self>, Receiver<Arc<PadStats>>) {
        let (retirement, retired) = std::sync::mpsc::sync_channel(max_pad_series);
        (
            Arc::new(Self {
                pads: papaya::HashMap::new(),
                registration: Mutex::new(RegistrationState {
                    active: 0,
                    next_id: 1,
                }),
                pipelines: RwLock::default(),
                queues: RwLock::default(),
                queue_cache: Mutex::default(),
                queue_identities_dirty: Arc::default(),
                include_filter,
                exclude_filter,
                max_pad_series,
                retirement,
                untracked_series_limit: AtomicU64::new(0),
                export_emit_errors: AtomicU64::new(0),
                export_flush_errors: AtomicU64::new(0),
                dropped_retirements: AtomicU64::new(0),
            }),
            retired,
        )
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
        let raw_element = element.path_string().to_string();
        let raw_pad = pad.name().to_string();
        let identity = format!("{raw_element}:{raw_pad}");
        let included = self.included(&identity);
        let labels = (
            sanitize_tag_value(&raw_element),
            sanitize_tag_value(&raw_pad),
        );

        let mut registration = self.registration.lock();
        let pads = self.pads.pin();
        if let Some(entry) = pads.get(&key) {
            Self::increment_entry(entry, buffers, bytes, &self.untracked_series_limit);
            return;
        }
        let entry = if !included {
            PadEntry::Ignored(IgnoreReason::Filtered)
        } else if registration.active >= self.max_pad_series {
            PadEntry::Ignored(IgnoreReason::SeriesLimit)
        } else if let Some(next_id) = registration.next_id.checked_add(1) {
            let id = registration.next_id;
            registration.next_id = next_id;
            registration.active += 1;
            PadEntry::Tracked(Arc::new(PadStats {
                id,
                element: labels.0,
                pad: labels.1,
                buffers: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
            }))
        } else {
            PadEntry::Ignored(IgnoreReason::SeriesIdOverflow)
        };
        let entry = pads.get_or_insert(key, entry);
        Self::increment_entry(entry, buffers, bytes, &self.untracked_series_limit);
    }

    fn increment_entry(
        entry: &PadEntry,
        buffers: u64,
        bytes: u64,
        untracked_series_limit: &AtomicU64,
    ) {
        match entry {
            PadEntry::Tracked(stats) => {
                stats.buffers.fetch_add(buffers, Ordering::Relaxed);
                stats.bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            PadEntry::Ignored(IgnoreReason::SeriesLimit | IgnoreReason::SeriesIdOverflow) => {
                untracked_series_limit.fetch_add(buffers, Ordering::Relaxed);
            }
            PadEntry::Ignored(IgnoreReason::Filtered) => {}
        }
    }

    pub(crate) fn active_pads(&self) -> Vec<Arc<PadStats>> {
        self.pads
            .pin()
            .iter()
            .filter_map(|(_key, entry)| match entry {
                PadEntry::Tracked(stats) => Some(Arc::clone(stats)),
                PadEntry::Ignored(_) => None,
            })
            .collect()
    }

    pub(crate) fn remove_pad(&self, pad: &gst::Pad) {
        self.remove_pad_key(pad.as_ptr() as usize);
    }

    pub(crate) fn remove_object_key(&self, key: usize) {
        self.invalidate_queue_identities();
        self.remove_pad_key(key);
        self.pipelines.write().remove(&key);
        self.remove_queue_key(key);
    }

    fn remove_pad_key(&self, key: usize) {
        let mut registration = self.registration.lock();
        let pads = self.pads.pin();
        if let Some(PadEntry::Tracked(stats)) = pads.remove(&key) {
            registration.active = registration.active.saturating_sub(1);
            if let Err(error) = self.retirement.try_send(Arc::clone(stats)) {
                match error {
                    TrySendError::Full(_) | TrySendError::Disconnected(_) => {
                        self.dropped_retirements.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    pub(crate) fn track_pipeline(&self, pipeline: &gst::Pipeline) {
        let raw = pipeline.name().to_string();
        if !self.included(&raw) {
            return;
        }
        self.pipelines
            .write()
            .entry(pipeline.as_ptr() as usize)
            .or_insert_with(|| PipelineEntry {
                pipeline: pipeline.downgrade(),
                name: sanitize_tag_value(&raw),
                observed_transition: AtomicBool::new(false),
            });
    }

    pub(crate) fn set_pipeline_state(&self, pipeline: &gst::Pipeline, _state: gst::State) {
        self.track_pipeline(pipeline);
        if let Some(entry) = self.pipelines.read().get(&(pipeline.as_ptr() as usize)) {
            entry.observed_transition.store(true, Ordering::Release);
        }
    }

    pub(crate) fn pipeline_snapshots(&self) -> Vec<PipelineSnapshot> {
        let snapshot = self
            .pipelines
            .read()
            .iter()
            .filter(|(_key, entry)| entry.observed_transition.load(Ordering::Acquire))
            .map(|(key, entry)| (*key, entry.pipeline.clone(), entry.name.clone()))
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(snapshot.len());
        let mut stale = Vec::new();
        for (key, weak, name) in snapshot {
            if let Some(pipeline) = weak.upgrade() {
                result.push(PipelineSnapshot {
                    pipeline: name,
                    state: pipeline.current_state(),
                });
            } else {
                stale.push(key);
            }
        }
        if !stale.is_empty() {
            let mut pipelines = self.pipelines.write();
            for key in stale {
                pipelines.remove(&key);
            }
        }
        result
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
                    labels: sanitize_tag_value(&identity),
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

    pub(crate) fn queue_snapshots(&self) -> Vec<QueueSnapshot> {
        let mut cache = self.queue_cache.lock();
        self.refresh_queue_cache(&mut cache);
        let mut result = Vec::with_capacity(cache.queues.len());
        let mut stale = Vec::new();
        for cached in &cache.queues {
            let Some(entry) = cached.entry.upgrade() else {
                continue;
            };
            if let Some(element) = entry.element.upgrade() {
                // Serialize refresh with notifications so a concurrent setter cannot lose invalidation.
                let limits = *entry.limits.lock().get_or_insert_with(|| QueueLimits {
                    buffers: element.property("max-size-buffers"),
                    bytes: element.property("max-size-bytes"),
                    time: element.property("max-size-time"),
                });
                result.push(QueueSnapshot {
                    element: cached.labels.clone(),
                    level_buffers: element.property("current-level-buffers"),
                    level_bytes: element.property("current-level-bytes"),
                    level_seconds: Duration::from_nanos(element.property("current-level-time"))
                        .as_secs_f64(),
                    capacity_buffers: limits.buffers,
                    capacity_bytes: limits.bytes,
                    capacity_seconds: Duration::from_nanos(limits.time).as_secs_f64(),
                });
            } else {
                stale.push(cached.key);
            }
        }
        for key in stale {
            self.remove_queue_key(key);
        }
        result
    }

    fn remove_queue_key(&self, key: usize) {
        // Disconnect handlers and release GStreamer references after unlocking the map.
        let entry = self.queues.write().remove(&key);
        if entry.is_some() {
            self.invalidate_queue_identities();
        }
        drop(entry);
    }
}

pub(crate) fn sanitize_tag_value(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len().min(MAX_TAG_VALUE_BYTES));
    for character in value.chars() {
        if sanitized.len() >= MAX_TAG_VALUE_BYTES {
            break;
        }
        let output =
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/') {
                character
            } else {
                '_'
            };
        sanitized.push(output);
    }
    if sanitized.is_empty() {
        sanitized.push('_');
    }
    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pad(element_name: &str) -> (gst::Element, gst::Pad) {
        gst::init().expect("initializing GStreamer");
        let element = gst::ElementFactory::make("identity")
            .name(element_name)
            .build()
            .expect("constructing identity");
        let pad = element.static_pad("src").expect("identity source pad");
        (element, pad)
    }

    #[test]
    fn metrics_filter_precedence_and_sanitization() {
        let (metrics, _retired) = Metrics::new(
            Some(Regex::new("included").expect("include regex")),
            Some(Regex::new("excluded").expect("exclude regex")),
            2,
        );
        let (_included, pad) = test_pad("included:name");
        metrics.update_pad(&pad, 2, 5);
        let (_excluded, excluded_pad) = test_pad("included-excluded");
        metrics.update_pad(&excluded_pad, 3, 7);
        let snapshot = metrics.active_pads();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].buffers.load(Ordering::Relaxed), 2);
        assert!(!snapshot[0].element.contains(':'));
        assert_eq!(sanitize_tag_value("a,b|c:d\n☃"), "a_b_c_d__");
        assert_eq!(sanitize_tag_value(""), "_");
        assert_eq!(sanitize_tag_value(&"x".repeat(300)).len(), 192);
    }

    #[test]
    fn metrics_cap_slot_reuse_and_retirement() {
        let (metrics, retired) = Metrics::new(None, None, 1);
        let (_first, first_pad) = test_pad("first");
        let (_second, second_pad) = test_pad("second");
        metrics.update_pad(&first_pad, 1, 10);
        metrics.update_pad(&second_pad, 2, 20);
        assert_eq!(metrics.active_pads().len(), 1);
        assert_eq!(metrics.untracked_series_limit.load(Ordering::Relaxed), 2);
        metrics.remove_pad(&first_pad);
        let removed = retired.try_recv().expect("retired tracked pad");
        assert_eq!(removed.buffers.load(Ordering::Relaxed), 1);
        let (_third, third_pad) = test_pad("third");
        metrics.update_pad(&third_pad, 3, 30);
        assert_eq!(metrics.active_pads().len(), 1);
        metrics.update_pad(&second_pad, 4, 40);
        assert_eq!(metrics.untracked_series_limit.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn concurrent_metrics_cap_is_exact() {
        let (metrics, _retired) = Metrics::new(None, None, 2);
        let elements = (0..8)
            .map(|index| test_pad(&format!("pad-{index}")))
            .collect::<Vec<_>>();
        std::thread::scope(|scope| {
            for (_element, pad) in &elements {
                let metrics = &metrics;
                scope.spawn(move || metrics.update_pad(pad, 1, 1));
            }
        });
        assert_eq!(metrics.active_pads().len(), 2);
        assert_eq!(metrics.untracked_series_limit.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn queue_metrics_follow_nested_bin_paths_and_filters() {
        gst::init().expect("initializing GStreamer");
        let include = Regex::new("GstPipeline:included").expect("include regex");
        let exclude = Regex::new("excluded").expect("exclude regex");
        for factory in ["queue", "queue2"] {
            for filtered in [false, true] {
                let (metrics, _retired) = Metrics::new(
                    filtered.then(|| include.clone()),
                    filtered.then(|| exclude.clone()),
                    1,
                );
                let has_queue =
                    |snapshots: &[QueueSnapshot], queue: &gst::Element, capacity: u32| {
                        snapshots.iter().any(|snapshot| {
                            snapshot.element == sanitize_tag_value(&queue.path_string())
                                && snapshot.capacity_buffers == capacity
                        })
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
                let _initial = metrics.queue_snapshots();
                let first_pipeline = gst::Pipeline::builder().name("included_first").build();
                let second_pipeline = gst::Pipeline::builder().name("included_second").build();
                let excluded = gst::Pipeline::builder().name("included_excluded").build();
                first_pipeline.add(&first_bin).expect("parenting first bin");
                second_pipeline
                    .add(&second_bin)
                    .expect("parenting second bin");

                metrics.invalidate_queue_identities();
                let snapshots = metrics.queue_snapshots();
                assert_eq!(snapshots.len(), 2);
                assert!(has_queue(&snapshots, &first, 11), "{snapshots:?}");
                assert!(has_queue(&snapshots, &second, 22), "{snapshots:?}");

                let old_name = sanitize_tag_value(&first.path_string());
                first_pipeline
                    .remove(&first_bin)
                    .expect("unparenting first bin");
                excluded
                    .add(&first_bin)
                    .expect("moving bin into excluded pipeline");
                metrics.invalidate_queue_identities();
                let snapshots = metrics.queue_snapshots();
                assert!(
                    snapshots
                        .iter()
                        .all(|snapshot| snapshot.element != old_name)
                );
                assert_eq!(
                    has_queue(&snapshots, &first, 11),
                    !filtered,
                    "{snapshots:?}"
                );
                assert!(has_queue(&snapshots, &second, 22), "{snapshots:?}");

                excluded
                    .remove(&first_bin)
                    .expect("unparenting excluded bin");
                first_pipeline
                    .add(&first_bin)
                    .expect("restoring included bin");
                second_bin.remove(&second).expect("removing second queue");
                metrics.remove_object_key(second.as_ptr() as usize);
                metrics.invalidate_queue_identities();
                let snapshots = metrics.queue_snapshots();
                assert_eq!(snapshots.len(), 1);
                assert!(has_queue(&snapshots, &first, 11), "{snapshots:?}");
            }
        }
    }

    #[test]
    fn queue_cache_handles_notifications_verification_and_cleanup() {
        gst::init().expect("initializing GStreamer");
        let (metrics, _retired) = Metrics::new(None, None, 1);
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
            let output = metrics.queue_snapshots().pop().expect("tracked queue");
            assert_eq!(output.element, sanitize_tag_value(&queue.path_string()));
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
        let before = metrics.queue_snapshots();
        let owner = gst::Bin::builder().name("owner").build();
        root.set_property("parent", &owner);
        assert_eq!(metrics.queue_snapshots(), before);
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
                let _output = metrics.queue_snapshots();
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
        let _output = metrics.queue_snapshots();
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
            let (metrics, _retired) = Metrics::new(None, None, 1);
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
                let snapshot = metrics.queue_snapshots().pop().expect("tracked queue");
                assert_eq!(snapshot.capacity_buffers, value);
                assert_eq!(snapshot.capacity_bytes, value * 1024);
                assert_eq!(
                    snapshot.capacity_seconds.to_bits(),
                    f64::from(value).to_bits()
                );
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
                    let _snapshot = metrics.queue_snapshots();
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
            let (metrics, _retired) = Metrics::new(None, None, 1);
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
                let snapshots = metrics.queue_snapshots();
                assert_eq!(snapshots.len(), queues.len());
                for (queue, capacity) in &queues {
                    let name = sanitize_tag_value(&queue.path_string());
                    let snapshot = snapshots
                        .iter()
                        .find(|snapshot| snapshot.element == name)
                        .expect("labels match the native GStreamer path");
                    assert_eq!(snapshot.capacity_buffers, capacity + offset);
                }
            }
        }
    }

    #[test]
    fn queue_tracker_drops_stale_weak_entries() {
        gst::init().expect("initializing GStreamer");
        let (metrics, _retired) = Metrics::new(None, None, 1);
        let queue = gst::ElementFactory::make("queue")
            .build()
            .expect("constructing queue");
        let weak = queue.downgrade();
        metrics.track_queue(&queue);
        assert_eq!(metrics.queue_snapshots().len(), 1);
        drop(queue);
        assert!(
            weak.upgrade().is_none(),
            "cache must not keep the queue alive"
        );
        assert!(metrics.queue_snapshots().is_empty());
    }

    #[test]
    fn retirement_queue_full_is_bounded() {
        let (metrics, _retired) = Metrics::new(None, None, 1);
        let (_first, first_pad) = test_pad("one");
        metrics.update_pad(&first_pad, 1, 1);
        metrics.remove_pad(&first_pad);
        let (_second, second_pad) = test_pad("two");
        metrics.update_pad(&second_pad, 1, 1);
        metrics.remove_pad(&second_pad);
        assert_eq!(metrics.dropped_retirements.load(Ordering::Relaxed), 1);
    }
}
