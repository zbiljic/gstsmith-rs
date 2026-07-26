use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use futures_util::StreamExt;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::base_src::CreateSuccess;
use gst_base::subclass::prelude::*;
use s2_sdk::types::{AppendRetryPolicy, ReadFrom, ReadInput, ReadStart, SequencedRecord};
use tokio_util::sync::CancellationToken;

use crate::config::{ConnectionSettings, ValidatedConnection, sanitized_error};
use crate::{meta, runtime};

#[derive(Clone, Copy, Debug, Default, Eq, glib::Enum, PartialEq)]
#[enum_type(name = "GstS2StartMode")]
enum StartMode {
    #[default]
    #[enum_value(name = "Earliest", nick = "earliest")]
    Earliest,
    #[enum_value(name = "Sequence", nick = "sequence")]
    Sequence,
    #[enum_value(name = "Timestamp", nick = "timestamp")]
    Timestamp,
    #[enum_value(name = "Tail offset", nick = "tail-offset")]
    TailOffset,
}

#[derive(Clone)]
struct Settings {
    connection: ConnectionSettings,
    caps: Option<gst::Caps>,
    start_mode: StartMode,
    start_seq_num: u64,
    start_timestamp: u64,
    tail_offset: u64,
    clamp_to_tail: bool,
    ignore_command_records: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            connection: ConnectionSettings::default(),
            caps: None,
            start_mode: StartMode::Earliest,
            start_seq_num: 0,
            start_timestamp: 0,
            tail_offset: 0,
            clamp_to_tail: false,
            ignore_command_records: false,
        }
    }
}

#[derive(Clone)]
struct WorkerConfig {
    connection: ValidatedConnection,
    start: ReadStart,
    ignore_command_records: bool,
}

struct Worker {
    receiver: Option<tokio::sync::mpsc::Receiver<SequencedRecord>>,
    handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
    failure: Arc<Mutex<Option<String>>>,
    error_posted: Arc<AtomicBool>,
}

#[derive(Default)]
enum State {
    #[default]
    Stopped,
    Started {
        worker: Worker,
        config: Box<WorkerConfig>,
        last_delivered: Option<u64>,
        flushing: bool,
    },
}

#[derive(Default)]
pub struct S2Src {
    settings: Mutex<Settings>,
    state: Mutex<State>,
}

impl S2Src {
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
            ["Invalid S2 source settings: {detail}"]
        )
    }

    fn post_read_error(&self, detail: &str, posted: &AtomicBool) {
        if !posted.swap(true, Ordering::Relaxed) {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Read,
                ["S2 source read failed: {detail}"]
            );
        }
    }

    fn buffer_from_record(
        &self,
        record: &SequencedRecord,
        basin: &str,
        stream: &str,
    ) -> Result<gst::Buffer, gst::FlowError> {
        let mut buffer = gst::Buffer::from_slice(record.body.clone());
        meta::attach_record(
            buffer.get_mut().ok_or(gst::FlowError::Error)?,
            basin,
            stream,
            record,
        )
        .map_err(|error| {
            gst::element_imp_error!(
                self,
                gst::StreamError::Format,
                ["Failed to attach S2 record metadata: {error}"]
            );
            gst::FlowError::Error
        })?;
        Ok(buffer)
    }
}

#[glib::object_subclass]
impl ObjectSubclass for S2Src {
    const NAME: &'static str = "GstSmithS2Src";
    type Type = super::S2Src;
    type ParentType = gst_base::PushSrc;
}

impl ObjectImpl for S2Src {
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
                glib::ParamSpecBoxed::builder::<gst::Caps>("caps")
                    .nick("Caps")
                    .blurb("Optional fixed output caps")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecEnum::builder::<StartMode>("start-mode")
                    .nick("Start Mode")
                    .blurb("S2 read start position kind")
                    .default_value(StartMode::Earliest)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("start-seq-num")
                    .nick("Start Sequence Number")
                    .blurb("Starting sequence number when start-mode is sequence")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("start-timestamp")
                    .nick("Start Timestamp")
                    .blurb("Starting timestamp when start-mode is timestamp")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("tail-offset")
                    .nick("Tail Offset")
                    .blurb("Number of records before tail when start-mode is tail-offset")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("clamp-to-tail")
                    .nick("Clamp To Tail")
                    .blurb("Clamp an unwritten start position to the current tail")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("ignore-command-records")
                    .nick("Ignore Command Records")
                    .blurb("Ask S2 to omit command records")
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
            "caps" => {
                if let Ok(caps) = value.get::<Option<gst::Caps>>() {
                    settings.caps = caps;
                }
            }
            "start-mode" => {
                if let Ok(mode) = value.get::<StartMode>() {
                    settings.start_mode = mode;
                }
            }
            "start-seq-num" => {
                if let Ok(seq_num) = value.get::<u64>() {
                    settings.start_seq_num = seq_num;
                }
            }
            "start-timestamp" => {
                if let Ok(timestamp) = value.get::<u64>() {
                    settings.start_timestamp = timestamp;
                }
            }
            "tail-offset" => {
                if let Ok(offset) = value.get::<u64>() {
                    settings.tail_offset = offset;
                }
            }
            "clamp-to-tail" => {
                if let Ok(clamp) = value.get::<bool>() {
                    settings.clamp_to_tail = clamp;
                }
            }
            "ignore-command-records" => {
                if let Ok(ignore) = value.get::<bool>() {
                    settings.ignore_command_records = ignore;
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
            "caps" => settings.caps.to_value(),
            "start-mode" => settings.start_mode.to_value(),
            "start-seq-num" => settings.start_seq_num.to_value(),
            "start-timestamp" => settings.start_timestamp.to_value(),
            "tail-offset" => settings.tail_offset.to_value(),
            "clamp-to-tail" => settings.clamp_to_tail.to_value(),
            "ignore-command-records" => settings.ignore_command_records.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for S2Src {}

impl ElementImpl for S2Src {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "S2 Source",
                "Source/Network",
                "Reads one S2 record per arbitrary byte buffer",
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
                .expect("constructing the s2src pad template"),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseSrcImpl for S2Src {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings().clone();
        let connection = settings
            .connection
            .validate(AppendRetryPolicy::NoSideEffects)
            .map_err(Self::settings_error)?;
        let start = configured_start(&settings);
        let config = WorkerConfig {
            connection,
            start,
            ignore_command_records: settings.ignore_command_records,
        };
        if let Some(caps) = settings.caps.as_ref() {
            self.obj()
                .set_caps(caps)
                .map_err(|error| Self::settings_error(format!("failed to set caps: {error}")))?;
        }
        let worker = spawn_worker(&config, self.obj().downgrade()).map_err(Self::settings_error)?;
        *self.state() = State::Started {
            worker,
            config: Box::new(config),
            last_delivered: None,
            flushing: false,
        };
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let old_state = std::mem::take(&mut *self.state());
        if let State::Started { worker, .. } = old_state {
            stop_worker(worker)?;
        }
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn unlock(&self) -> Result<(), gst::ErrorMessage> {
        if let State::Started {
            worker, flushing, ..
        } = &mut *self.state()
        {
            *flushing = true;
            worker.cancel.cancel();
        }
        Ok(())
    }

    fn unlock_stop(&self) -> Result<(), gst::ErrorMessage> {
        let (old_worker, config, resume) = {
            let mut state = self.state();
            let State::Started {
                worker,
                config,
                last_delivered,
                flushing,
            } = &mut *state
            else {
                return Ok(());
            };
            *flushing = false;
            let resume = last_delivered
                .map(|seq_num| {
                    seq_num
                        .checked_add(1)
                        .ok_or_else(|| Self::settings_error("S2 sequence number exhausted"))
                })
                .transpose()?;
            let replacement = spawn_worker_with_start(
                config,
                resume.map(ReadFrom::SeqNum),
                self.obj().downgrade(),
            )
            .map_err(Self::settings_error)?;
            (
                std::mem::replace(worker, replacement),
                config.clone(),
                resume,
            )
        };
        stop_worker(old_worker)?;
        let _ = (config, resume);
        Ok(())
    }
}

impl PushSrcImpl for S2Src {
    fn create(
        &self,
        _buffer: Option<&mut gst::BufferRef>,
    ) -> Result<CreateSuccess, gst::FlowError> {
        let (mut receiver, cancel) = {
            let mut state = self.state();
            match &mut *state {
                State::Started {
                    worker, flushing, ..
                } => {
                    if *flushing {
                        return Err(gst::FlowError::Flushing);
                    }
                    (
                        worker.receiver.take().ok_or(gst::FlowError::Error)?,
                        worker.cancel.clone(),
                    )
                }
                State::Stopped => return Err(gst::FlowError::Flushing),
            }
        };

        let runtime = runtime::runtime().map_err(|error| {
            gst::element_imp_error!(self, gst::ResourceError::Failed, ["{error}"]);
            gst::FlowError::Error
        })?;
        let record = runtime.block_on(async {
            tokio::select! {
                () = cancel.cancelled() => None,
                record = receiver.recv() => record,
            }
        });

        let (flushing, failure, posted) = {
            let mut state = self.state();
            let State::Started {
                worker, flushing, ..
            } = &mut *state
            else {
                return Err(gst::FlowError::Flushing);
            };
            worker.receiver = Some(receiver);
            (
                *flushing,
                worker
                    .failure
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone(),
                Arc::clone(&worker.error_posted),
            )
        };
        if flushing || cancel.is_cancelled() {
            return Err(gst::FlowError::Flushing);
        }
        let Some(record) = record else {
            let detail = failure.unwrap_or_else(|| "S2 read worker closed unexpectedly".to_owned());
            self.post_read_error(&detail, &posted);
            return Err(gst::FlowError::Error);
        };
        let settings = self.settings();
        let basin = settings.connection.basin.as_deref().ok_or_else(|| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Settings,
                ["S2 basin disappeared while the source was running"]
            );
            gst::FlowError::Error
        })?;
        let stream = settings.connection.stream.as_deref().ok_or_else(|| {
            gst::element_imp_error!(
                self,
                gst::ResourceError::Settings,
                ["S2 stream disappeared while the source was running"]
            );
            gst::FlowError::Error
        })?;
        let buffer = self.buffer_from_record(&record, basin, stream)?;
        {
            let mut state = self.state();
            let State::Started {
                worker,
                last_delivered,
                flushing,
                ..
            } = &mut *state
            else {
                return Err(gst::FlowError::Flushing);
            };
            commit_delivered_cursor(
                Some(last_delivered),
                *flushing,
                &worker.cancel,
                &cancel,
                record.seq_num,
            )?;
        }
        Ok(CreateSuccess::NewBuffer(buffer))
    }
}

fn commit_delivered_cursor(
    last_delivered: Option<&mut Option<u64>>,
    flushing: bool,
    worker_cancel: &CancellationToken,
    create_cancel: &CancellationToken,
    seq_num: u64,
) -> Result<(), gst::FlowError> {
    if flushing || worker_cancel != create_cancel || create_cancel.is_cancelled() {
        return Err(gst::FlowError::Flushing);
    }
    let Some(last_delivered) = last_delivered else {
        return Err(gst::FlowError::Flushing);
    };
    *last_delivered = Some(seq_num);
    Ok(())
}

fn configured_start(settings: &Settings) -> ReadStart {
    let from = match settings.start_mode {
        StartMode::Earliest => ReadFrom::SeqNum(0),
        StartMode::Sequence => ReadFrom::SeqNum(settings.start_seq_num),
        StartMode::Timestamp => ReadFrom::Timestamp(settings.start_timestamp),
        StartMode::TailOffset => ReadFrom::TailOffset(settings.tail_offset),
    };
    ReadStart::new()
        .with_from(from)
        .with_clamp_to_tail(settings.clamp_to_tail)
}

fn spawn_worker(
    config: &WorkerConfig,
    weak: glib::WeakRef<super::S2Src>,
) -> Result<Worker, String> {
    spawn_worker_with_start(config, None, weak)
}

fn spawn_worker_with_start(
    config: &WorkerConfig,
    start_override: Option<ReadFrom>,
    weak: glib::WeakRef<super::S2Src>,
) -> Result<Worker, String> {
    let runtime = runtime::runtime()?;
    let (sender, receiver) = tokio::sync::mpsc::channel(config.connection.queue_capacity);
    let cancel = CancellationToken::new();
    let failure = Arc::new(Mutex::new(None));
    let error_posted = Arc::new(AtomicBool::new(false));
    let worker_config = config.clone();
    let worker_cancel = cancel.clone();
    let worker_failure = Arc::clone(&failure);
    let worker_posted = Arc::clone(&error_posted);
    let handle = runtime.spawn(async move {
        let input = ReadInput::new()
            .with_start(match start_override {
                Some(from) => ReadStart::new().with_from(from),
                None => worker_config.start,
            })
            .with_ignore_command_records(worker_config.ignore_command_records);
        let result = read_worker(worker_config.connection, input, sender, &worker_cancel).await;
        if let Err(detail) = result {
            *worker_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(detail.clone());
            if !worker_posted.swap(true, Ordering::Relaxed)
                && let Some(element) = weak.upgrade()
            {
                gst::element_error!(
                    element,
                    gst::ResourceError::Read,
                    ["S2 source read failed: {detail}"]
                );
            }
        }
    });
    Ok(Worker {
        receiver: Some(receiver),
        handle,
        cancel,
        failure,
        error_posted,
    })
}

async fn read_worker(
    connection: ValidatedConnection,
    input: ReadInput,
    sender: tokio::sync::mpsc::Sender<SequencedRecord>,
    cancel: &CancellationToken,
) -> Result<(), String> {
    let s2 = s2_sdk::S2::new(connection.s2).map_err(|error| sanitized_error(&error))?;
    let stream = s2.basin(connection.basin).stream(connection.stream);
    if !input.start.clamp_to_tail {
        let tail = tokio::select! {
            () = cancel.cancelled() => return Ok(()),
            result = stream.check_tail() => {
                result.map_err(|error| sanitized_error(&error))?
            }
        };
        validate_start_against_tail(input.start.from, tail.seq_num, tail.timestamp)?;
    }
    let mut session = tokio::select! {
        () = cancel.cancelled() => return Ok(()),
        result = stream.read_session(input) => {
            result.map_err(|error| sanitized_error(&error))?
        }
    };
    loop {
        let batch = tokio::select! {
            () = cancel.cancelled() => return Ok(()),
            batch = session.next() => batch,
        };
        let Some(batch) = batch else {
            return Err("S2 read session ended unexpectedly".to_owned());
        };
        let batch = batch.map_err(|error| sanitized_error(&error))?;
        for record in batch.records {
            tokio::select! {
                () = cancel.cancelled() => return Ok(()),
                result = sender.send(record) => {
                    if result.is_err() {
                        return Ok(());
                    }
                }
            }
        }
    }
}

fn validate_start_against_tail(
    start: ReadFrom,
    tail_seq_num: u64,
    tail_timestamp: u64,
) -> Result<(), String> {
    match start {
        ReadFrom::SeqNum(seq_num) if seq_num > tail_seq_num => Err(format!(
            "S2 read start sequence {seq_num} is beyond current tail {tail_seq_num}"
        )),
        ReadFrom::Timestamp(timestamp) if timestamp > tail_timestamp => Err(format!(
            "S2 read start timestamp {timestamp} is beyond current tail timestamp {tail_timestamp}"
        )),
        _ => Ok(()),
    }
}

fn stop_worker(mut worker: Worker) -> Result<(), gst::ErrorMessage> {
    worker.cancel.cancel();
    worker.receiver.take();
    let runtime = runtime::runtime().map_err(S2Src::settings_error)?;
    if runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(2), &mut worker.handle).await })
        .is_err()
    {
        worker.handle.abort();
        let _join_result = runtime.block_on(worker.handle);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(from: ReadFrom) -> (&'static str, u64) {
        match from {
            ReadFrom::SeqNum(value) => ("sequence", value),
            ReadFrom::Timestamp(value) => ("timestamp", value),
            ReadFrom::TailOffset(value) => ("tail-offset", value),
        }
    }

    #[test]
    fn configured_start_maps_every_public_mode() {
        let cases = [
            (StartMode::Earliest, ReadFrom::SeqNum(0)),
            (StartMode::Sequence, ReadFrom::SeqNum(7)),
            (StartMode::Timestamp, ReadFrom::Timestamp(11)),
            (StartMode::TailOffset, ReadFrom::TailOffset(13)),
        ];
        for (mode, expected) in cases {
            let settings = Settings {
                start_mode: mode,
                start_seq_num: 7,
                start_timestamp: 11,
                tail_offset: 13,
                clamp_to_tail: true,
                ..Settings::default()
            };
            let start = configured_start(&settings);
            assert!(start.clamp_to_tail);
            assert_eq!(parts(start.from), parts(expected));
        }
    }

    #[test]
    fn sequence_resume_detects_exhaustion() {
        let cancel = CancellationToken::new();
        let mut last_delivered = None;
        commit_delivered_cursor(Some(&mut last_delivered), false, &cancel, &cancel, u64::MAX)
            .expect("a maximum sequence number is a valid delivered cursor");
        assert_eq!(last_delivered, Some(u64::MAX));
        assert_eq!(
            last_delivered.and_then(|seq_num| seq_num.checked_add(1)),
            None
        );
    }

    #[test]
    fn delivery_resume_cursor_commits_only_for_active_attempts() {
        let active = CancellationToken::new();
        let mut last_delivered = Some(3);
        commit_delivered_cursor(Some(&mut last_delivered), false, &active, &active, 4)
            .expect("an active create attempt commits its delivered sequence");
        assert_eq!(last_delivered, Some(4));

        let flushing = CancellationToken::new();
        assert_eq!(
            commit_delivered_cursor(Some(&mut last_delivered), true, &flushing, &flushing, 5,),
            Err(gst::FlowError::Flushing)
        );
        assert_eq!(last_delivered, Some(4));

        let canceled = CancellationToken::new();
        canceled.cancel();
        assert_eq!(
            commit_delivered_cursor(Some(&mut last_delivered), false, &canceled, &canceled, 5,),
            Err(gst::FlowError::Flushing)
        );
        assert_eq!(last_delivered, Some(4));

        let replacement = CancellationToken::new();
        assert_eq!(
            commit_delivered_cursor(Some(&mut last_delivered), false, &replacement, &active, 5,),
            Err(gst::FlowError::Flushing)
        );
        assert_eq!(last_delivered, Some(4));

        let stopped = CancellationToken::new();
        assert_eq!(
            commit_delivered_cursor(None, false, &stopped, &stopped, 5),
            Err(gst::FlowError::Flushing)
        );
        assert_eq!(last_delivered, Some(4));
    }

    #[test]
    fn unclamped_start_preflight_rejects_unwritten_positions() {
        validate_start_against_tail(ReadFrom::SeqNum(8), 7, 11)
            .expect_err("an unwritten sequence must fail");
        validate_start_against_tail(ReadFrom::Timestamp(12), 7, 11)
            .expect_err("an unwritten timestamp must fail");
        validate_start_against_tail(ReadFrom::SeqNum(7), 7, 11)
            .expect("the current tail is a valid starting position");
        validate_start_against_tail(ReadFrom::Timestamp(11), 7, 11)
            .expect("the current tail timestamp is a valid starting position");
        validate_start_against_tail(ReadFrom::TailOffset(u64::MAX), 7, 11)
            .expect("tail offsets are relative and cannot be unwritten");
    }
}
