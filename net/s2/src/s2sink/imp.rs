use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use s2_sdk::batching::BatchingConfig;
use s2_sdk::producer::{ProducerConfig, RecordSubmitTicket};
use s2_sdk::types::{AppendRecord, AppendRetryPolicy};
use tokio_util::sync::CancellationToken;

use crate::config::{
    ConnectionSettings, SinkAppendRetryPolicy, ValidatedConnection, load_fencing_token,
    sanitized_error,
};
use crate::{meta, runtime};

const DEFAULT_BATCH_LINGER: u64 = 5_000_000;
const DEFAULT_BATCH_MAX_RECORDS: u32 = 1_000;
const DEFAULT_BATCH_MAX_BYTES: u32 = 1_048_576;
const DEFAULT_MAX_UNACKED_BYTES: u32 = 5_242_880;
const DEFAULT_SHUTDOWN_TIMEOUT: u64 = 10_000_000_000;

#[derive(Clone)]
struct Settings {
    connection: ConnectionSettings,
    batch_linger: u64,
    batch_max_records: u32,
    batch_max_bytes: u32,
    max_unacked_bytes: u32,
    append_retry_policy: SinkAppendRetryPolicy,
    fencing_token_file: Option<String>,
    match_seq_num_enabled: bool,
    match_seq_num: u64,
    preserve_timestamp: bool,
    shutdown_timeout: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            connection: ConnectionSettings::default(),
            batch_linger: DEFAULT_BATCH_LINGER,
            batch_max_records: DEFAULT_BATCH_MAX_RECORDS,
            batch_max_bytes: DEFAULT_BATCH_MAX_BYTES,
            max_unacked_bytes: DEFAULT_MAX_UNACKED_BYTES,
            append_retry_policy: SinkAppendRetryPolicy::NoSideEffects,
            fencing_token_file: None,
            match_seq_num_enabled: false,
            match_seq_num: 0,
            preserve_timestamp: false,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
        }
    }
}

struct QueueState {
    records: VecDeque<AppendRecord>,
    accepting: bool,
    flushing: bool,
    failure: Option<String>,
    done: bool,
}

struct SharedQueue {
    state: Mutex<QueueState>,
    capacity_available: Condvar,
    done: Condvar,
    worker_wakeup: tokio::sync::Notify,
    capacity: usize,
    cancel: CancellationToken,
    error_posted: AtomicBool,
}

impl SharedQueue {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                records: VecDeque::with_capacity(capacity),
                accepting: true,
                flushing: false,
                failure: None,
                done: false,
            }),
            capacity_available: Condvar::new(),
            done: Condvar::new(),
            worker_wakeup: tokio::sync::Notify::new(),
            capacity,
            cancel: CancellationToken::new(),
            error_posted: AtomicBool::new(false),
        }
    }

    fn state(&self) -> MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn close(&self) {
        self.state().accepting = false;
        self.capacity_available.notify_all();
        self.worker_wakeup.notify_one();
    }

    fn store_failure(&self, detail: String) {
        let mut state = self.state();
        if state.failure.is_none() {
            state.failure = Some(detail);
        }
        state.accepting = false;
        drop(state);
        self.capacity_available.notify_all();
        self.done.notify_all();
        self.worker_wakeup.notify_one();
    }

    fn claim_error_post(&self) -> bool {
        !self.error_posted.swap(true, Ordering::Relaxed)
    }

    fn finish(&self) {
        self.state().done = true;
        self.capacity_available.notify_all();
        self.done.notify_all();
    }
}

#[derive(Debug, Eq, PartialEq)]
enum EnqueueError {
    Flushing,
    Terminal(String),
    Closed,
}

fn enqueue_record(shared: &SharedQueue, record: AppendRecord) -> Result<(), EnqueueError> {
    enqueue_record_with_wait_hook(shared, record, || {})
}

fn enqueue_record_with_wait_hook(
    shared: &SharedQueue,
    record: AppendRecord,
    wait_hook: impl FnOnce(),
) -> Result<(), EnqueueError> {
    let mut wait_hook = Some(wait_hook);
    let mut state = shared.state();
    loop {
        if state.flushing {
            return Err(EnqueueError::Flushing);
        }
        if let Some(detail) = state.failure.as_ref() {
            return Err(EnqueueError::Terminal(detail.clone()));
        }
        if !state.accepting {
            return Err(EnqueueError::Closed);
        }
        if state.records.len() < shared.capacity {
            state.records.push_back(record);
            drop(state);
            shared.worker_wakeup.notify_one();
            return Ok(());
        }
        if let Some(wait_hook) = wait_hook.take() {
            wait_hook();
        }
        state = shared
            .capacity_available
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

struct WorkerConfig {
    connection: ValidatedConnection,
    producer: ProducerConfig,
}

#[derive(Debug, Eq, PartialEq)]
enum DrainOutcome {
    Complete,
    Flushing,
    Timeout(String),
    WorkerFailure(String),
}

#[derive(Default)]
enum State {
    #[default]
    Stopped,
    Started {
        shared: Arc<SharedQueue>,
        worker: tokio::task::JoinHandle<()>,
        shutdown_timeout: Duration,
    },
}

#[derive(Default)]
pub struct S2Sink {
    settings: Mutex<Settings>,
    state: Mutex<State>,
}

impl S2Sink {
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
            ["Invalid S2 sink settings: {detail}"]
        )
    }

    fn render_error(&self, detail: impl std::fmt::Display) -> gst::FlowError {
        gst::element_imp_error!(
            self,
            gst::ResourceError::Write,
            ["Failed to append an S2 record: {detail}"]
        );
        gst::FlowError::Error
    }

    fn drain(shared: &SharedQueue, timeout: Duration) -> DrainOutcome {
        Self::drain_with_wait_hook(shared, timeout, || {})
    }

    fn drain_with_wait_hook(
        shared: &SharedQueue,
        timeout: Duration,
        wait_hook: impl FnOnce(),
    ) -> DrainOutcome {
        let mut wait_hook = Some(wait_hook);
        shared.close();
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            return DrainOutcome::Timeout("shutdown timeout cannot be represented".to_owned());
        };
        let mut state = shared.state();
        while !state.done && !state.flushing && state.failure.is_none() {
            let now = Instant::now();
            if now >= deadline {
                drop(state);
                shared.cancel.cancel();
                let detail =
                    "S2 sink shutdown timed out; accepted records may be unconfirmed".to_owned();
                shared.store_failure(detail.clone());
                return DrainOutcome::Timeout(detail);
            }
            let remaining = deadline.saturating_duration_since(now);
            if let Some(wait_hook) = wait_hook.take() {
                wait_hook();
            }
            let waited = shared
                .done
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = waited.0;
        }
        if let Some(detail) = state.failure.as_ref() {
            DrainOutcome::WorkerFailure(detail.clone())
        } else if state.done {
            DrainOutcome::Complete
        } else if state.flushing {
            DrainOutcome::Flushing
        } else {
            DrainOutcome::WorkerFailure("S2 sink worker did not finish draining".to_owned())
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for S2Sink {
    const NAME: &'static str = "GstSmithS2Sink";
    type Type = super::S2Sink;
    type ParentType = gst_base::BaseSink;
}

impl ObjectImpl for S2Sink {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_sync(false);
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let mut properties = ConnectionSettings::property_specs();
            properties.extend([
                glib::ParamSpecUInt64::builder("batch-linger")
                    .nick("Batch Linger")
                    .blurb("Maximum producer batching linger in nanoseconds")
                    .default_value(DEFAULT_BATCH_LINGER)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("batch-max-records")
                    .nick("Batch Maximum Records")
                    .blurb("Maximum S2 records per append batch")
                    .minimum(1)
                    .maximum(DEFAULT_BATCH_MAX_RECORDS)
                    .default_value(DEFAULT_BATCH_MAX_RECORDS)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("batch-max-bytes")
                    .nick("Batch Maximum Bytes")
                    .blurb("Maximum metered bytes per S2 append batch")
                    .minimum(8)
                    .maximum(DEFAULT_BATCH_MAX_BYTES)
                    .default_value(DEFAULT_BATCH_MAX_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-unacked-bytes")
                    .nick("Maximum Unacknowledged Bytes")
                    .blurb("Maximum metered bytes awaiting S2 acknowledgement")
                    .minimum(DEFAULT_BATCH_MAX_BYTES)
                    .default_value(DEFAULT_MAX_UNACKED_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecEnum::builder::<SinkAppendRetryPolicy>("append-retry-policy")
                    .nick("Append Retry Policy")
                    .blurb("Whether ambiguous appends may be retried")
                    .default_value(SinkAppendRetryPolicy::NoSideEffects)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("fencing-token-file")
                    .nick("Fencing Token File")
                    .blurb("Optional file containing an S2 fencing token")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("match-seq-num-enabled")
                    .nick("Match Sequence Number Enabled")
                    .blurb("Apply the initial append sequence precondition")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("match-seq-num")
                    .nick("Match Sequence Number")
                    .blurb("Initial append sequence precondition")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("preserve-timestamp")
                    .nick("Preserve Timestamp")
                    .blurb("Copy GstS2RecordMeta timestamp to appended records")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("shutdown-timeout")
                    .nick("Shutdown Timeout")
                    .blurb("Maximum EOS and state-stop drain time in nanoseconds")
                    .minimum(1)
                    .default_value(DEFAULT_SHUTDOWN_TIMEOUT)
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
            "batch-linger" => set(&mut settings.batch_linger, value),
            "batch-max-records" => set(&mut settings.batch_max_records, value),
            "batch-max-bytes" => set(&mut settings.batch_max_bytes, value),
            "max-unacked-bytes" => set(&mut settings.max_unacked_bytes, value),
            "append-retry-policy" => set(&mut settings.append_retry_policy, value),
            "fencing-token-file" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    settings.fencing_token_file = path.filter(|path| !path.is_empty());
                }
            }
            "match-seq-num-enabled" => set(&mut settings.match_seq_num_enabled, value),
            "match-seq-num" => set(&mut settings.match_seq_num, value),
            "preserve-timestamp" => set(&mut settings.preserve_timestamp, value),
            "shutdown-timeout" => set(&mut settings.shutdown_timeout, value),
            _ => {}
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings();
        if let Some(value) = settings.connection.property(pspec) {
            return value;
        }
        match pspec.name() {
            "batch-linger" => settings.batch_linger.to_value(),
            "batch-max-records" => settings.batch_max_records.to_value(),
            "batch-max-bytes" => settings.batch_max_bytes.to_value(),
            "max-unacked-bytes" => settings.max_unacked_bytes.to_value(),
            "append-retry-policy" => settings.append_retry_policy.to_value(),
            "fencing-token-file" => settings.fencing_token_file.to_value(),
            "match-seq-num-enabled" => settings.match_seq_num_enabled.to_value(),
            "match-seq-num" => settings.match_seq_num.to_value(),
            "preserve-timestamp" => settings.preserve_timestamp.to_value(),
            "shutdown-timeout" => settings.shutdown_timeout.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for S2Sink {}

impl ElementImpl for S2Sink {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "S2 Sink",
                "Sink/Network",
                "Appends one arbitrary byte buffer per S2 record",
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
                .expect("constructing the s2sink pad template"),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for S2Sink {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        let append_policy: AppendRetryPolicy = settings.append_retry_policy.into();
        let connection = settings
            .connection
            .validate(append_policy)
            .map_err(Self::settings_error)?;
        let batching = BatchingConfig::new()
            .with_linger(Duration::from_nanos(settings.batch_linger))
            .with_max_batch_records(
                usize::try_from(settings.batch_max_records).map_err(Self::settings_error)?,
            )
            .map_err(Self::settings_error)?
            .with_max_batch_bytes(
                usize::try_from(settings.batch_max_bytes).map_err(Self::settings_error)?,
            )
            .map_err(Self::settings_error)?;
        let mut producer = ProducerConfig::new()
            .with_batching(batching)
            .with_max_unacked_bytes(settings.max_unacked_bytes)
            .map_err(Self::settings_error)?;
        if let Some(token) = load_fencing_token(settings.fencing_token_file.as_deref())
            .map_err(Self::settings_error)?
        {
            producer = producer.with_fencing_token(token);
        }
        if settings.match_seq_num_enabled {
            producer = producer.with_match_seq_num(settings.match_seq_num);
        }
        let shared = Arc::new(SharedQueue::new(connection.queue_capacity));
        let runtime = runtime::runtime().map_err(Self::settings_error)?;
        let weak = self.obj().downgrade();
        let worker_shared = Arc::clone(&shared);
        let worker = runtime.spawn(async move {
            let result = append_worker(
                WorkerConfig {
                    connection,
                    producer,
                },
                Arc::clone(&worker_shared),
            )
            .await;
            if let Err(detail) = result {
                worker_shared.store_failure(detail.clone());
                if worker_shared.claim_error_post()
                    && let Some(element) = weak.upgrade()
                {
                    gst::element_error!(
                        element,
                        gst::ResourceError::Write,
                        ["S2 sink worker failed: {detail}"]
                    );
                }
            }
            worker_shared.finish();
        });
        *self.state() = State::Started {
            shared,
            worker,
            shutdown_timeout: Duration::from_nanos(settings.shutdown_timeout),
        };
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let old_state = std::mem::take(&mut *self.state());
        if let State::Started {
            shared,
            mut worker,
            shutdown_timeout,
        } = old_state
        {
            let drain_result = Self::drain(&shared, shutdown_timeout);
            let runtime = runtime::runtime().map_err(Self::settings_error)?;
            if runtime
                .block_on(async { tokio::time::timeout(Duration::from_secs(1), &mut worker).await })
                .is_err()
            {
                worker.abort();
                let _join_result = runtime.block_on(worker);
            }
            if let DrainOutcome::Timeout(detail) | DrainOutcome::WorkerFailure(detail) =
                drain_result
            {
                return Err(gst::error_msg!(gst::ResourceError::Write, ["{detail}"]));
            }
        }
        Ok(())
    }

    fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.settings().clone();
        let envelope = if meta::is_present(buffer) {
            Some(meta::read(buffer).map_err(|error| self.render_error(error))?)
        } else {
            None
        };
        let map = buffer
            .map_readable()
            .map_err(|error| self.render_error(format!("failed to map input buffer: {error}")))?;
        let mut record =
            AppendRecord::new(map.as_slice().to_vec()).map_err(|error| self.render_error(error))?;
        if let Some(envelope) = envelope.as_ref() {
            record = record
                .with_headers(
                    meta::regular_headers(envelope).map_err(|error| self.render_error(error))?,
                )
                .map_err(|error| self.render_error(error))?;
            if settings.preserve_timestamp {
                record = record.with_timestamp(envelope.timestamp);
            }
        }

        let shared = {
            let state = self.state();
            let State::Started { shared, .. } = &*state else {
                return Err(gst::FlowError::Flushing);
            };
            Arc::clone(shared)
        };
        match enqueue_record(&shared, record) {
            Ok(()) => Ok(gst::FlowSuccess::Ok),
            Err(EnqueueError::Flushing | EnqueueError::Closed) => Err(gst::FlowError::Flushing),
            Err(EnqueueError::Terminal(_detail)) => Err(gst::FlowError::Error),
        }
    }

    fn event(&self, event: gst::Event) -> bool {
        if matches!(event.view(), gst::EventView::Eos(_)) {
            let (shared, timeout) = {
                let state = self.state();
                let State::Started {
                    shared,
                    shutdown_timeout,
                    ..
                } = &*state
                else {
                    return self.parent_event(event);
                };
                (Arc::clone(shared), *shutdown_timeout)
            };
            match Self::drain(&shared, timeout) {
                DrainOutcome::Complete | DrainOutcome::Flushing => {}
                DrainOutcome::Timeout(detail) | DrainOutcome::WorkerFailure(detail) => {
                    if shared.claim_error_post() {
                        gst::element_imp_error!(
                            self,
                            gst::ResourceError::Write,
                            ["S2 sink EOS durability barrier failed: {detail}"]
                        );
                    }
                    return false;
                }
            }
        }
        self.parent_event(event)
    }

    fn unlock(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { shared, .. } = &*self.state() {
            let mut state = shared.state();
            state.flushing = true;
            drop(state);
            shared.capacity_available.notify_all();
            shared.done.notify_all();
        }
        Ok(())
    }

    fn unlock_stop(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started { shared, .. } = &*self.state() {
            shared.state().flushing = false;
        }
        Ok(())
    }
}

async fn append_worker(config: WorkerConfig, shared: Arc<SharedQueue>) -> Result<(), String> {
    let s2 = s2_sdk::S2::new(config.connection.s2).map_err(|error| sanitized_error(&error))?;
    let stream = s2
        .basin(config.connection.basin)
        .stream(config.connection.stream);
    let producer = stream.producer(config.producer);
    let mut tickets = FuturesUnordered::<RecordSubmitTicket>::new();

    loop {
        let (record, accepting) = {
            let mut state = shared.state();
            let record = state.records.pop_front();
            let accepting = state.accepting;
            if record.is_some() {
                shared.capacity_available.notify_one();
            }
            (record, accepting)
        };
        if let Some(record) = record {
            let submission = async {
                producer
                    .submit(record)
                    .await
                    .map_err(|error| sanitized_error(&error))
            };
            let ticket =
                submit_while_observing(submission, &mut tickets, &shared.cancel, |error| {
                    sanitized_error(&error)
                })
                .await?;
            tickets.push(ticket);
            continue;
        }
        if !accepting {
            break;
        }
        if tickets.is_empty() {
            tokio::select! {
                () = shared.cancel.cancelled() => {
                    return Err("S2 sink worker was cancelled".to_owned());
                }
                () = shared.worker_wakeup.notified() => {}
            }
        } else {
            tokio::select! {
                () = shared.cancel.cancelled() => {
                    return Err("S2 sink worker was cancelled with unconfirmed records".to_owned());
                }
                () = shared.worker_wakeup.notified() => {}
                result = tickets.next() => {
                    if let Some(result) = result {
                        result.map_err(|error| sanitized_error(&error))?;
                    }
                }
            }
        }
    }

    tokio::select! {
        () = shared.cancel.cancelled() => {
            return Err("S2 sink drain was cancelled with unconfirmed records".to_owned());
        }
        result = producer.close() => {
            result.map_err(|error| sanitized_error(&error))?;
        }
    }
    while let Some(result) = tickets.next().await {
        result.map_err(|error| sanitized_error(&error))?;
    }
    Ok(())
}

async fn submit_while_observing<T, F, A, E>(
    submission: impl Future<Output = Result<T, String>>,
    tickets: &mut FuturesUnordered<F>,
    cancel: &CancellationToken,
    map_ticket_error: impl Fn(E) -> String,
) -> Result<T, String>
where
    F: Future<Output = Result<A, E>>,
{
    tokio::pin!(submission);
    loop {
        if tickets.is_empty() {
            return tokio::select! {
                () = cancel.cancelled() => {
                    Err("S2 sink worker was cancelled with unconfirmed records".to_owned())
                }
                result = &mut submission => result,
            };
        }
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                return Err("S2 sink worker was cancelled with unconfirmed records".to_owned());
            }
            result = tickets.next() => {
                if let Some(result) = result {
                    result.map_err(&map_ticket_error)?;
                }
            }
            result = &mut submission => return result,
        }
    }
}

fn set<T: Copy + for<'a> glib::value::FromValue<'a>>(slot: &mut T, value: &glib::Value) {
    if let Ok(new_value) = value.get::<T>() {
        *slot = new_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(value: u8) -> AppendRecord {
        AppendRecord::new(vec![value]).expect("valid test record")
    }

    fn filled_queue() -> Arc<SharedQueue> {
        let shared = Arc::new(SharedQueue::new(1));
        shared.state().records.push_back(record(0));
        shared
    }

    fn blocked_enqueue(
        shared: &Arc<SharedQueue>,
    ) -> (
        std::sync::mpsc::Receiver<Result<(), EnqueueError>>,
        std::thread::JoinHandle<()>,
    ) {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let thread_shared = Arc::clone(shared);
        let handle = std::thread::spawn(move || {
            result_tx
                .send(enqueue_record_with_wait_hook(
                    &thread_shared,
                    record(1),
                    move || ready_tx.send(()).expect("reporting blocked enqueue"),
                ))
                .expect("reporting enqueue result");
        });
        ready_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("enqueue thread starts");
        (result_rx, handle)
    }

    fn assert_enqueue(
        receiver: &std::sync::mpsc::Receiver<Result<(), EnqueueError>>,
        handle: std::thread::JoinHandle<()>,
        expected: &Result<(), EnqueueError>,
    ) {
        let result = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("blocked enqueue wakes");
        handle.join().expect("enqueue thread joins");
        assert_eq!(&result, expected);
    }

    #[test]
    fn full_queue_wakes_for_capacity_flush_failure_and_close() {
        let capacity = filled_queue();
        let (receiver, handle) = blocked_enqueue(&capacity);
        let _removed = capacity.state().records.pop_front();
        capacity.capacity_available.notify_one();
        assert_enqueue(&receiver, handle, &Ok(()));

        let flushing = filled_queue();
        let (receiver, handle) = blocked_enqueue(&flushing);
        flushing.state().flushing = true;
        flushing.capacity_available.notify_all();
        assert_enqueue(&receiver, handle, &Err(EnqueueError::Flushing));

        let failed = filled_queue();
        let (receiver, handle) = blocked_enqueue(&failed);
        failed.store_failure("terminal ticket failure".to_owned());
        assert_enqueue(
            &receiver,
            handle,
            &Err(EnqueueError::Terminal("terminal ticket failure".to_owned())),
        );
        assert!(failed.claim_error_post());
        assert!(!failed.claim_error_post());

        let closed = filled_queue();
        let (receiver, handle) = blocked_enqueue(&closed);
        closed.close();
        assert_enqueue(&receiver, handle, &Err(EnqueueError::Closed));
    }

    #[test]
    fn shutdown_timeout_is_bounded_and_marks_unconfirmed_failure() {
        let shared = SharedQueue::new(1);
        let started = Instant::now();
        let DrainOutcome::Timeout(error) = S2Sink::drain(&shared, Duration::from_millis(10)) else {
            panic!("a worker that never finishes must time out");
        };
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(error.contains("unconfirmed"));
        assert!(shared.cancel.is_cancelled());
        assert!(
            shared
                .state()
                .failure
                .as_deref()
                .is_some_and(|detail| detail.contains("timed out"))
        );
    }

    #[test]
    fn drain_wakes_for_flush_without_recording_a_failure() {
        let shared = Arc::new(SharedQueue::new(1));
        let wait_barrier = Arc::new(std::sync::Barrier::new(2));
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let thread_shared = Arc::clone(&shared);
        let thread_barrier = Arc::clone(&wait_barrier);
        let handle = std::thread::spawn(move || {
            result_tx
                .send(S2Sink::drain_with_wait_hook(
                    &thread_shared,
                    Duration::from_secs(5),
                    move || {
                        thread_barrier.wait();
                    },
                ))
                .expect("reporting drain result");
        });

        wait_barrier.wait();
        shared.state().flushing = true;
        shared.done.notify_all();

        let result = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("drain wakes for flush");
        handle.join().expect("drain thread joins");
        assert_eq!(result, DrainOutcome::Flushing);
        assert!(shared.state().failure.is_none());
        assert!(!shared.cancel.is_cancelled());
        assert!(!shared.error_posted.load(Ordering::Relaxed));
    }

    #[test]
    fn drain_reports_worker_failure_before_flush() {
        let shared = SharedQueue::new(1);
        {
            let mut state = shared.state();
            state.failure = Some("terminal ticket failure".to_owned());
            state.flushing = true;
        }

        assert_eq!(
            S2Sink::drain(&shared, Duration::from_secs(1)),
            DrainOutcome::WorkerFailure("terminal ticket failure".to_owned())
        );
    }

    #[test]
    fn ready_ticket_failure_preempts_another_submission() {
        let mut tickets = FuturesUnordered::new();
        tickets.push(std::future::ready(Err::<(), _>(
            "terminal acknowledgement failure".to_owned(),
        )));
        let submission_polled = Arc::new(AtomicBool::new(false));
        let submission_flag = Arc::clone(&submission_polled);
        let submission = async move {
            submission_flag.store(true, Ordering::Relaxed);
            Ok::<_, String>(())
        };
        let cancel = CancellationToken::new();
        let error = runtime::runtime()
            .expect("test runtime")
            .block_on(submit_while_observing(
                submission,
                &mut tickets,
                &cancel,
                std::convert::identity,
            ))
            .expect_err("ready acknowledgement failure is terminal");
        assert_eq!(error, "terminal acknowledgement failure");
        assert!(!submission_polled.load(Ordering::Relaxed));
    }
}
