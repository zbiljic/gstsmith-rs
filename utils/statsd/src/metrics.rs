use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

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

struct PipelineEntry {
    pipeline: glib::WeakRef<gst::Pipeline>,
    name: String,
    observed_transition: AtomicBool,
}

struct QueueEntry {
    element: glib::WeakRef<gst::Element>,
    name: String,
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
    queues: RwLock<HashMap<usize, QueueEntry>>,
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
        self.remove_pad_key(key);
        self.pipelines.write().remove(&key);
        self.queues.write().remove(&key);
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
        let raw = element.path_string().to_string();
        if !self.included(&raw) {
            return;
        }
        self.queues
            .write()
            .entry(element.as_ptr() as usize)
            .or_insert_with(|| QueueEntry {
                element: element.downgrade(),
                name: sanitize_tag_value(&raw),
            });
    }

    pub(crate) fn queue_snapshots(&self) -> Vec<QueueSnapshot> {
        let snapshot = self
            .queues
            .read()
            .iter()
            .map(|(key, entry)| (*key, entry.element.clone(), entry.name.clone()))
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(snapshot.len());
        let mut stale = Vec::new();
        for (key, weak, name) in snapshot {
            if let Some(element) = weak.upgrade() {
                result.push(QueueSnapshot {
                    element: name,
                    level_buffers: element.property("current-level-buffers"),
                    level_bytes: element.property("current-level-bytes"),
                    level_seconds: Duration::from_nanos(element.property("current-level-time"))
                        .as_secs_f64(),
                    capacity_buffers: element.property("max-size-buffers"),
                    capacity_bytes: element.property("max-size-bytes"),
                    capacity_seconds: Duration::from_nanos(element.property("max-size-time"))
                        .as_secs_f64(),
                });
            } else {
                stale.push(key);
            }
        }
        if !stale.is_empty() {
            let mut queues = self.queues.write();
            for key in stale {
                queues.remove(&key);
            }
        }
        result
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
