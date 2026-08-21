use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, LazyLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cadence::prelude::{Counted, Gauged};
use cadence::{BufferedUdpMetricSink, StatsdClient};

use crate::metrics::{Metrics, PadStats};

const BUFFER_CAPACITY: usize = 1_432;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(2);

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "statsd",
        gst::DebugColorFlags::empty(),
        Some("StatsD metrics tracer worker"),
    )
});

#[derive(Clone, Debug)]
pub(crate) struct WorkerConfig {
    pub(crate) destination: SocketAddr,
    pub(crate) prefix: String,
    pub(crate) global_tags: Vec<(String, String)>,
    pub(crate) flush_interval: Duration,
}

pub(crate) struct WorkerHandle {
    shutdown: Sender<()>,
    thread: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl WorkerHandle {
    pub(crate) fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub(crate) fn stop(&mut self) -> Result<(), String> {
        if self.thread.is_none() {
            self.running.store(false, Ordering::Release);
            return Ok(());
        }
        if self.shutdown.send(()).is_err() && self.is_running() {
            gst::warning!(CAT, "StatsD worker shutdown receiver closed unexpectedly");
        }
        let result = self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_panic_payload| "StatsD worker thread terminated unexpectedly".to_owned())
        });
        self.running.store(false, Ordering::Release);
        result
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            gst::error!(CAT, "Failed to stop StatsD worker: {error}");
        }
    }
}

pub(crate) fn start(
    config: WorkerConfig,
    metrics: Arc<Metrics>,
    retired: Receiver<Arc<PadStats>>,
) -> Result<WorkerHandle, String> {
    let (shutdown, shutdown_rx) = std::sync::mpsc::channel();
    let (ready, ready_rx) = std::sync::mpsc::sync_channel(1);
    let running = Arc::new(AtomicBool::new(false));
    let thread_running = Arc::clone(&running);
    let thread = thread::Builder::new()
        .name("gst-statsd-export".to_owned())
        .spawn(move || {
            run(
                &config,
                &metrics,
                &retired,
                &shutdown_rx,
                &ready,
                &thread_running,
            );
        })
        .map_err(|error| format!("failed to start StatsD worker thread: {error}"))?;

    match ready_rx.recv_timeout(STARTUP_TIMEOUT) {
        Ok(Ok(())) => Ok(WorkerHandle {
            shutdown,
            thread: Some(thread),
            running,
        }),
        Ok(Err(error)) => join_startup_failure(thread, error),
        Err(error) => {
            let message = format!("StatsD worker startup handshake failed: {error}");
            if shutdown.send(()).is_err() {
                gst::warning!(CAT, "StatsD worker exited before startup shutdown signal");
            }
            join_startup_failure(thread, message)
        }
    }
}

fn join_startup_failure(thread: JoinHandle<()>, error: String) -> Result<WorkerHandle, String> {
    match thread.join() {
        Ok(()) => Err(error),
        Err(_panic_payload) => Err(format!("{error}; StatsD worker did not join cleanly")),
    }
}

struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn run(
    config: &WorkerConfig,
    metrics: &Metrics,
    retired: &Receiver<Arc<PadStats>>,
    shutdown: &Receiver<()>,
    ready: &std::sync::mpsc::SyncSender<Result<(), String>>,
    running: &Arc<AtomicBool>,
) {
    let local = match config.destination.ip() {
        IpAddr::V4(_) => "0.0.0.0:0",
        IpAddr::V6(_) => "[::]:0",
    };
    let socket = match UdpSocket::bind(local) {
        Ok(socket) => socket,
        Err(error) => {
            send_startup_error(ready, format!("failed to bind StatsD UDP socket: {error}"));
            return;
        }
    };
    if let Err(error) = socket.set_nonblocking(true) {
        send_startup_error(
            ready,
            format!("failed to make StatsD UDP socket nonblocking: {error}"),
        );
        return;
    }
    let sink =
        match BufferedUdpMetricSink::with_capacity(config.destination, socket, BUFFER_CAPACITY) {
            Ok(sink) => sink,
            Err(error) => {
                send_startup_error(ready, format!("failed to construct StatsD sink: {error}"));
                return;
            }
        };
    let mut builder = StatsdClient::builder(&config.prefix, sink);
    for (key, value) in &config.global_tags {
        builder = builder.with_tag(key, value);
    }
    let client = builder.build();
    running.store(true, Ordering::Release);
    let _running_guard = RunningGuard(Arc::clone(running));
    if ready.send(Ok(())).is_err() {
        return;
    }

    let mut state = ExportState::default();
    let mut error_log = ErrorLog::default();
    loop {
        match shutdown.recv_timeout(config.flush_interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                export_snapshot(&client, metrics, retired, &mut state, &mut error_log);
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                export_snapshot(&client, metrics, retired, &mut state, &mut error_log);
            }
        }
    }
}

fn send_startup_error(ready: &std::sync::mpsc::SyncSender<Result<(), String>>, error: String) {
    if ready.send(Err(error)).is_err() {
        gst::warning!(CAT, "StatsD startup receiver closed before worker failure");
    }
}

#[derive(Default)]
struct ExportState {
    pads: HashMap<u64, PadCursor>,
    untracked: CounterCursor,
    emit_errors: CounterCursor,
    flush_errors: CounterCursor,
    dropped: CounterCursor,
}

#[derive(Default)]
struct PadCursor {
    buffers: CounterCursor,
    bytes: CounterCursor,
}

#[derive(Default)]
struct CounterCursor {
    value: u64,
    was_active: bool,
}

#[derive(Default)]
struct ErrorLog {
    failing: bool,
    last_log: Option<Instant>,
}

impl ErrorLog {
    fn failure(&mut self, message: &str) {
        let now = Instant::now();
        if !self.failing
            || self
                .last_log
                .is_none_or(|last| now.duration_since(last) >= Duration::from_mins(1))
        {
            gst::warning!(CAT, "{message}");
            self.last_log = Some(now);
        }
        self.failing = true;
    }

    fn recovery(&mut self) {
        if self.failing {
            gst::info!(CAT, "StatsD export recovered after a previous failure");
            self.failing = false;
            self.last_log = None;
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one snapshot keeps pad cursors, lifecycle gauges, diagnostics, and the flush order visible"
)]
fn export_snapshot(
    client: &StatsdClient,
    metrics: &Metrics,
    retired: &Receiver<Arc<PadStats>>,
    state: &mut ExportState,
    error_log: &mut ErrorLog,
) {
    let active = metrics.active_pads();
    export_pad_snapshot(client, metrics, &active, retired, state, error_log);

    for pipeline in metrics.pipeline_snapshots() {
        let current = match pipeline.state {
            gst::State::Null => "null",
            gst::State::Ready => "ready",
            gst::State::Paused => "paused",
            gst::State::Playing => "playing",
            gst::State::VoidPending => continue,
        };
        for state_name in ["null", "ready", "paused", "playing"] {
            let value = u64::from(state_name == current);
            let result = client
                .gauge_with_tags("gstreamer.pipeline.state", value)
                .with_tag("pipeline", &pipeline.pipeline)
                .with_tag("state", state_name)
                .try_send();
            record_emit_result(result.is_ok(), metrics, error_log);
        }
    }

    for queue in metrics.queue_snapshots() {
        emit_queue_gauge(
            client,
            metrics,
            error_log,
            "gstreamer.queue.level_buffers",
            u64::from(queue.level_buffers),
            &queue.element,
        );
        emit_queue_gauge(
            client,
            metrics,
            error_log,
            "gstreamer.queue.level_bytes",
            u64::from(queue.level_bytes),
            &queue.element,
        );
        emit_queue_gauge_f64(
            client,
            metrics,
            error_log,
            "gstreamer.queue.level_seconds",
            queue.level_seconds,
            &queue.element,
        );
        emit_queue_gauge(
            client,
            metrics,
            error_log,
            "gstreamer.queue.capacity_buffers",
            u64::from(queue.capacity_buffers),
            &queue.element,
        );
        emit_queue_gauge(
            client,
            metrics,
            error_log,
            "gstreamer.queue.capacity_bytes",
            u64::from(queue.capacity_bytes),
            &queue.element,
        );
        emit_queue_gauge_f64(
            client,
            metrics,
            error_log,
            "gstreamer.queue.capacity_seconds",
            queue.capacity_seconds,
            &queue.element,
        );
    }

    export_internal(
        client,
        "gstreamer.untracked_pad_events",
        "series_limit",
        &metrics.untracked_series_limit,
        &mut state.untracked,
        metrics,
        error_log,
        InternalFailurePolicy::CountAndLog,
    );
    export_internal(
        client,
        "gstreamer.statsd_export_errors",
        "emit",
        &metrics.export_emit_errors,
        &mut state.emit_errors,
        metrics,
        error_log,
        InternalFailurePolicy::SuppressRecursion,
    );
    export_internal(
        client,
        "gstreamer.statsd_export_errors",
        "flush",
        &metrics.export_flush_errors,
        &mut state.flush_errors,
        metrics,
        error_log,
        InternalFailurePolicy::SuppressRecursion,
    );
    export_internal(
        client,
        "gstreamer.statsd_dropped_series",
        "retirement_queue_full",
        &metrics.dropped_retirements,
        &mut state.dropped,
        metrics,
        error_log,
        InternalFailurePolicy::CountAndLog,
    );

    match client.flush() {
        Ok(()) => error_log.recovery(),
        Err(error) => {
            metrics.export_flush_errors.fetch_add(1, Ordering::Relaxed);
            error_log.failure(&format!("StatsD flush failed: {error}"));
        }
    }
}

fn export_pad_snapshot(
    client: &StatsdClient,
    metrics: &Metrics,
    active: &[Arc<PadStats>],
    retired: &Receiver<Arc<PadStats>>,
    state: &mut ExportState,
    error_log: &mut ErrorLog,
) {
    let active_ids = active.iter().map(|pad| pad.id).collect::<HashSet<_>>();
    let mut pads = active
        .iter()
        .map(|pad| (pad.id, Arc::clone(pad)))
        .collect::<HashMap<_, _>>();
    let mut retired_ids = HashSet::new();
    while let Ok(pad) = retired.try_recv() {
        retired_ids.insert(pad.id);
        // A retirement can arrive after the active snapshot was collected. Replacing its active
        // entry makes the retirement value authoritative without emitting the series twice.
        pads.insert(pad.id, pad);
    }

    for pad in pads.values() {
        export_pad(client, metrics, pad, state, error_log);
    }
    for id in &retired_ids {
        // Retirement export is best effort: cursor state must not outlive the retired series even
        // when Cadence rejects its final emission.
        state.pads.remove(id);
    }
    state
        .pads
        .retain(|id, _cursor| active_ids.contains(id) && !retired_ids.contains(id));
}

fn export_pad(
    client: &StatsdClient,
    metrics: &Metrics,
    pad: &PadStats,
    state: &mut ExportState,
    error_log: &mut ErrorLog,
) {
    let cursor = state.pads.entry(pad.id).or_default();
    export_pad_counter(
        client,
        metrics,
        error_log,
        "gstreamer.pad.push_buffers",
        pad.buffers.load(Ordering::Relaxed),
        &pad.element,
        &pad.pad,
        &mut cursor.buffers,
    );
    export_pad_counter(
        client,
        metrics,
        error_log,
        "gstreamer.pad.push_bytes",
        pad.bytes.load(Ordering::Relaxed),
        &pad.element,
        &pad.pad,
        &mut cursor.bytes,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the arguments make cursor advancement and the two bounded tags explicit"
)]
fn export_pad_counter(
    client: &StatsdClient,
    metrics: &Metrics,
    error_log: &mut ErrorLog,
    key: &str,
    current: u64,
    element: &str,
    pad: &str,
    cursor: &mut CounterCursor,
) {
    let delta = current.wrapping_sub(cursor.value);
    if delta == 0 && !cursor.was_active {
        return;
    }
    let amount = delta.min(i64::MAX as u64);
    let result = client
        .count_with_tags(key, i64::try_from(amount).unwrap_or(i64::MAX))
        .with_tag("element", element)
        .with_tag("pad", pad)
        .try_send();
    if result.is_ok() {
        cursor.value = cursor.value.wrapping_add(amount);
        cursor.was_active = delta != 0;
    } else {
        metrics.export_emit_errors.fetch_add(1, Ordering::Relaxed);
        error_log.failure("StatsD counter emission failed");
    }
}

#[derive(Clone, Copy)]
enum InternalFailurePolicy {
    CountAndLog,
    SuppressRecursion,
}

#[expect(
    clippy::too_many_arguments,
    reason = "diagnostic recursion policy and its shared counter/error state must be explicit"
)]
fn export_internal(
    client: &StatsdClient,
    key: &str,
    reason: &str,
    counter: &AtomicU64,
    cursor: &mut CounterCursor,
    metrics: &Metrics,
    error_log: &mut ErrorLog,
    failure_policy: InternalFailurePolicy,
) {
    let current = counter.load(Ordering::Relaxed);
    let delta = current.wrapping_sub(cursor.value);
    if delta == 0 && !cursor.was_active {
        return;
    }
    let amount = delta.min(i64::MAX as u64);
    let accepted = client
        .count_with_tags(key, i64::try_from(amount).unwrap_or(i64::MAX))
        .with_tag("reason", reason)
        .try_send()
        .is_ok();
    if accepted {
        cursor.value = cursor.value.wrapping_add(amount);
        cursor.was_active = delta != 0;
    } else if matches!(failure_policy, InternalFailurePolicy::CountAndLog) {
        metrics.export_emit_errors.fetch_add(1, Ordering::Relaxed);
        error_log.failure("StatsD internal counter emission failed");
    }
}

fn emit_queue_gauge(
    client: &StatsdClient,
    metrics: &Metrics,
    error_log: &mut ErrorLog,
    key: &str,
    value: u64,
    element: &str,
) {
    let success = client
        .gauge_with_tags(key, value)
        .with_tag("element", element)
        .try_send()
        .is_ok();
    record_emit_result(success, metrics, error_log);
}

fn emit_queue_gauge_f64(
    client: &StatsdClient,
    metrics: &Metrics,
    error_log: &mut ErrorLog,
    key: &str,
    value: f64,
    element: &str,
) {
    let success = client
        .gauge_with_tags(key, value)
        .with_tag("element", element)
        .try_send()
        .is_ok();
    record_emit_result(success, metrics, error_log);
}

fn record_emit_result(success: bool, metrics: &Metrics, error_log: &mut ErrorLog) {
    if !success {
        metrics.export_emit_errors.fetch_add(1, Ordering::Relaxed);
        error_log.failure("StatsD metric emission failed");
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;
    use cadence::MetricSink;
    use gst::prelude::*;

    fn receiver() -> (UdpSocket, SocketAddr) {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("binding UDP receiver");
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("setting receiver timeout");
        let address = socket.local_addr().expect("receiver address");
        (socket, address)
    }

    #[test]
    fn worker_shutdown_flush_delivers_pending_counter_and_stops_within_budget() {
        gst::init().expect("initializing GStreamer");
        let (socket, destination) = receiver();
        let (metrics, retired) = Metrics::new(None, None, 1);
        let element = gst::ElementFactory::make("identity")
            .name("shutdown-observed")
            .build()
            .expect("identity");
        let pad = element.static_pad("src").expect("source pad");
        metrics.update_pad(&pad, 5, 50);
        let mut worker = start(
            WorkerConfig {
                destination,
                prefix: "gstsmith".to_owned(),
                global_tags: vec![("env".to_owned(), "test".to_owned())],
                flush_interval: Duration::from_mins(1),
            },
            Arc::clone(&metrics),
            retired,
        )
        .expect("starting worker");
        assert!(worker.is_running());
        let started = Instant::now();
        worker.stop().expect("stopping worker");
        worker.stop().expect("stopping worker twice");
        let mut buffer = [0_u8; 1_500];
        let size = socket.recv(&mut buffer).expect("receiving final flush");
        let payload = String::from_utf8_lossy(&buffer[..size]);
        assert!(payload.lines().any(|line| {
            line.starts_with("gstsmith.gstreamer.pad.push_buffers:5|c|")
                && line.contains("element:/GstIdentity_shutdown-observed")
                && line.contains("pad:src")
        }));
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(!worker.is_running());
    }

    #[test]
    fn export_counter_delta_and_idle_zero() {
        gst::init().expect("initializing GStreamer");
        let (socket, destination) = receiver();
        let (metrics, retired) = Metrics::new(None, None, 1);
        let element = gst::ElementFactory::make("identity")
            .name("observed")
            .build()
            .expect("identity");
        let pad = element.static_pad("src").expect("source pad");
        metrics.update_pad(&pad, 3, 9);
        let mut worker = start(
            WorkerConfig {
                destination,
                prefix: "gstsmith".to_owned(),
                global_tags: vec![("env".to_owned(), "test".to_owned())],
                flush_interval: Duration::from_millis(100),
            },
            Arc::clone(&metrics),
            retired,
        )
        .expect("starting worker");
        let mut payload = String::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline && !payload.contains("push_buffers:0|c") {
            let mut buffer = [0_u8; 1_500];
            if let Ok(size) = socket.recv(&mut buffer) {
                payload.push_str(&String::from_utf8_lossy(&buffer[..size]));
            }
        }
        worker.stop().expect("stopping worker");
        assert!(payload.contains("gstsmith.gstreamer.pad.push_buffers:3|c"));
        assert!(payload.contains("gstsmith.gstreamer.pad.push_buffers:0|c"));
        assert!(payload.contains("|#") && payload.contains("element:"));
        assert!(payload.contains("pad:src"));
        assert!(payload.contains("env:test"));
    }

    #[test]
    fn export_worst_case_line_fits_buffer() {
        let prefix = "p".repeat(128);
        let tag = "x".repeat(crate::metrics::MAX_TAG_VALUE_BYTES);
        let global_tags = (0..16)
            .map(|index| format!("k{index:02}:{}", "v".repeat(27)))
            .collect::<Vec<_>>()
            .join(",");
        let line = format!(
            "{prefix}.gstreamer.pad.push_bytes:{}|c|#element:{tag},pad:{tag},{global_tags}\n",
            i64::MAX,
        );
        assert!(
            line.len() <= BUFFER_CAPACITY,
            "wire line was {} bytes",
            line.len()
        );
    }

    #[derive(Debug)]
    struct FailOnceSink(AtomicBool);

    impl MetricSink for FailOnceSink {
        fn emit(&self, metric: &str) -> io::Result<usize> {
            if self.0.swap(false, Ordering::AcqRel) {
                Err(io::Error::other("injected immediate rejection"))
            } else {
                Ok(metric.len())
            }
        }
    }

    #[derive(Debug, Clone)]
    struct RecordingSink(Arc<std::sync::Mutex<Vec<String>>>);

    impl MetricSink for RecordingSink {
        fn emit(&self, metric: &str) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(metric.to_owned());
            Ok(metric.len())
        }
    }

    #[test]
    fn retirement_combines_post_snapshot_increment_into_one_final_delta() {
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = StatsdClient::from_sink("gstsmith", RecordingSink(Arc::clone(&recorded)));
        let (metrics, _unused_retired) = Metrics::new(None, None, 2);
        let active_pad = Arc::new(PadStats {
            id: 41,
            element: "element".to_owned(),
            pad: "src".to_owned(),
            buffers: AtomicU64::new(2),
            bytes: AtomicU64::new(0),
        });
        let active = vec![active_pad];
        let mut state = ExportState {
            pads: HashMap::from([(
                41,
                PadCursor {
                    buffers: CounterCursor {
                        value: 1,
                        was_active: true,
                    },
                    bytes: CounterCursor::default(),
                },
            )]),
            ..ExportState::default()
        };
        let mut error_log = ErrorLog::default();

        // Model the retirement handoff carrying a later value than the already captured active
        // snapshot. Duplicate handoffs remain one authoritative final export.
        let retired_pad = Arc::new(PadStats {
            id: 41,
            element: "element".to_owned(),
            pad: "src".to_owned(),
            buffers: AtomicU64::new(3),
            bytes: AtomicU64::new(0),
        });
        let (sender, retired) = std::sync::mpsc::sync_channel(2);
        sender
            .send(Arc::clone(&retired_pad))
            .expect("first retirement");
        sender.send(retired_pad).expect("duplicate retirement");
        export_pad_snapshot(
            &client,
            &metrics,
            &active,
            &retired,
            &mut state,
            &mut error_log,
        );

        let lines = recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains("gstreamer.pad.push_buffers:2|c"))
                .count(),
            1,
            "the final delta must include both increments since the prior cursor: {lines:?}"
        );
        assert_eq!(lines.len(), 1, "the series must emit once: {lines:?}");
        assert!(!state.pads.contains_key(&41));
    }

    #[test]
    fn retirement_without_post_snapshot_increment_does_not_emit_idle_zero() {
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = StatsdClient::from_sink("gstsmith", RecordingSink(Arc::clone(&recorded)));
        let (metrics, _unused_retired) = Metrics::new(None, None, 1);
        let pad = Arc::new(PadStats {
            id: 42,
            element: "element".to_owned(),
            pad: "src".to_owned(),
            buffers: AtomicU64::new(2),
            bytes: AtomicU64::new(0),
        });
        let active = vec![Arc::clone(&pad)];
        let mut state = ExportState {
            pads: HashMap::from([(
                pad.id,
                PadCursor {
                    buffers: CounterCursor {
                        value: 1,
                        was_active: true,
                    },
                    bytes: CounterCursor::default(),
                },
            )]),
            ..ExportState::default()
        };
        let mut error_log = ErrorLog::default();
        let (sender, retired) = std::sync::mpsc::sync_channel(1);
        sender.send(pad).expect("retirement");

        export_pad_snapshot(
            &client,
            &metrics,
            &active,
            &retired,
            &mut state,
            &mut error_log,
        );

        let lines = recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(lines.len(), 1, "the series must emit once: {lines:?}");
        assert!(
            lines[0].contains("gstreamer.pad.push_buffers:1|c"),
            "the single emission must contain the pending delta: {lines:?}"
        );
        assert!(
            lines.iter().all(|line| !line.contains("push_buffers:0|c")),
            "retirement must not synthesize a same-cycle idle zero: {lines:?}"
        );
        assert!(!state.pads.contains_key(&42));
    }

    #[test]
    fn retirement_failed_final_emit_does_not_leak_cursor() {
        let client = StatsdClient::from_sink("gstsmith", FailOnceSink(AtomicBool::new(true)));
        let (metrics, _unused_retired) = Metrics::new(None, None, 1);
        let pad = Arc::new(PadStats {
            id: 43,
            element: "element".to_owned(),
            pad: "src".to_owned(),
            buffers: AtomicU64::new(1),
            bytes: AtomicU64::new(0),
        });
        let mut state = ExportState::default();
        let mut error_log = ErrorLog::default();
        let (sender, retired) = std::sync::mpsc::sync_channel(1);
        sender.send(pad).expect("retirement");

        export_pad_snapshot(&client, &metrics, &[], &retired, &mut state, &mut error_log);

        assert_eq!(metrics.export_emit_errors.load(Ordering::Relaxed), 1);
        assert!(error_log.failing);
        assert!(!state.pads.contains_key(&43));
    }

    #[test]
    fn export_internal_failure_policy_counts_without_recursive_amplification() {
        let (metrics, _retired) = Metrics::new(None, None, 1);
        metrics.untracked_series_limit.store(1, Ordering::Relaxed);
        let ordinary_client =
            StatsdClient::from_sink("gstsmith", FailOnceSink(AtomicBool::new(true)));
        let mut ordinary_cursor = CounterCursor::default();
        let mut ordinary_log = ErrorLog::default();
        export_internal(
            &ordinary_client,
            "gstreamer.untracked_pad_events",
            "series_limit",
            &metrics.untracked_series_limit,
            &mut ordinary_cursor,
            &metrics,
            &mut ordinary_log,
            InternalFailurePolicy::CountAndLog,
        );
        assert_eq!(metrics.export_emit_errors.load(Ordering::Relaxed), 1);
        assert_eq!(ordinary_cursor.value, 0);
        assert!(ordinary_log.failing);

        let diagnostic_client =
            StatsdClient::from_sink("gstsmith", FailOnceSink(AtomicBool::new(true)));
        let mut diagnostic_cursor = CounterCursor::default();
        let mut diagnostic_log = ErrorLog::default();
        export_internal(
            &diagnostic_client,
            "gstreamer.statsd_export_errors",
            "emit",
            &metrics.export_emit_errors,
            &mut diagnostic_cursor,
            &metrics,
            &mut diagnostic_log,
            InternalFailurePolicy::SuppressRecursion,
        );
        assert_eq!(metrics.export_emit_errors.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostic_cursor.value, 0);
        assert!(!diagnostic_log.failing);
    }

    #[test]
    fn export_cursor_retries_after_immediate_rejection() {
        let client = StatsdClient::from_sink("gstsmith", FailOnceSink(AtomicBool::new(true)));
        let (metrics, _retired) = Metrics::new(None, None, 1);
        let mut cursor = CounterCursor::default();
        let mut error_log = ErrorLog::default();
        export_pad_counter(
            &client,
            &metrics,
            &mut error_log,
            "gstreamer.pad.push_buffers",
            7,
            "element",
            "src",
            &mut cursor,
        );
        assert_eq!(cursor.value, 0);
        export_pad_counter(
            &client,
            &metrics,
            &mut error_log,
            "gstreamer.pad.push_buffers",
            7,
            "element",
            "src",
            &mut cursor,
        );
        assert_eq!(cursor.value, 7);
        assert_eq!(metrics.export_emit_errors.load(Ordering::Relaxed), 1);
    }
}
