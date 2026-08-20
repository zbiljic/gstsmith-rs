use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;

use crate::backend::{self, OcrError};
use crate::message;
use crate::worker::{self, Counters, Job, Worker};

const DEFAULT_INTERVAL: u64 = 500_000_000;
const DEFAULT_MODEL_BYTES: u64 = 134_217_728;
const DEFAULT_FRAME_BYTES: u64 = 33_554_432;
const DEFAULT_MAX_LINES: u32 = 128;
const DEFAULT_TEXT_LENGTH: u32 = 512;
const MAX_MODEL_BYTES: u64 = 1_073_741_824;
const MAX_FRAME_BYTES: u64 = 134_217_728;

#[derive(Clone)]
struct Settings {
    detection_model: Option<PathBuf>,
    recognition_model: Option<PathBuf>,
    alphabet_file: Option<PathBuf>,
    allowed_characters: Option<String>,
    analysis_interval: u64,
    max_model_bytes: u64,
    max_frame_bytes: u64,
    max_lines: u32,
    max_text_length: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            detection_model: None,
            recognition_model: None,
            alphabet_file: None,
            allowed_characters: None,
            analysis_interval: DEFAULT_INTERVAL,
            max_model_bytes: DEFAULT_MODEL_BYTES,
            max_frame_bytes: DEFAULT_FRAME_BYTES,
            max_lines: DEFAULT_MAX_LINES,
            max_text_length: DEFAULT_TEXT_LENGTH,
        }
    }
}

#[derive(Default)]
enum Selection {
    #[default]
    Empty,
    SelectedWithoutPts,
    SelectedAt(gst::ClockTime),
}

#[derive(Default)]
struct Sampler {
    selection: Selection,
    last_seen_valid_pts: Option<gst::ClockTime>,
}

pub struct OcrsAnalysis {
    settings: Mutex<Settings>,
    pub(super) state: Mutex<Option<Worker>>,
    video_info: Mutex<Option<gst_video::VideoInfo>>,
    sampler: Mutex<Sampler>,
    counters: Arc<Counters>,
    generation: AtomicU64,
    next_id: AtomicU64,
    no_pts_warned: AtomicBool,
    worker_failure_reported: AtomicBool,
    #[cfg(test)]
    test_backend_factory: Mutex<Option<TestBackendFactory>>,
    #[cfg(test)]
    test_post_admission_hook: Mutex<Option<worker::PostAdmissionHook>>,
}

#[cfg(test)]
type TestBackendFactory =
    Arc<dyn Fn() -> Result<Box<dyn backend::OcrBackend>, OcrError> + Send + Sync>;

impl Default for OcrsAnalysis {
    fn default() -> Self {
        Self {
            settings: Mutex::new(Settings::default()),
            state: Mutex::new(None),
            video_info: Mutex::new(None),
            sampler: Mutex::new(Sampler::default()),
            counters: Arc::new(Counters {
                submitted: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                failed: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
            }),
            generation: AtomicU64::new(1),
            next_id: AtomicU64::new(1),
            no_pts_warned: AtomicBool::new(false),
            worker_failure_reported: AtomicBool::new(false),
            #[cfg(test)]
            test_backend_factory: Mutex::new(None),
            #[cfg(test)]
            test_post_admission_hook: Mutex::new(None),
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for OcrsAnalysis {
    const NAME: &'static str = "GstSmithOcrsAnalysis";
    type Type = super::OcrsAnalysis;
    type ParentType = gst_base::BaseTransform;
}

impl OcrsAnalysis {
    #[cfg(test)]
    pub(crate) fn set_test_backend_factory(&self, factory: TestBackendFactory) {
        if let Ok(mut slot) = self.test_backend_factory.lock() {
            *slot = Some(factory);
        }
    }
    #[cfg(test)]
    pub(crate) fn set_test_post_admission_hook(&self, hook: Option<worker::PostAdmissionHook>) {
        if let Ok(mut slot) = self.test_post_admission_hook.lock() {
            *slot = hook;
        }
    }
    fn reset_run(&self) {
        self.counters.reset();
        self.generation.store(1, Ordering::Relaxed);
        self.next_id.store(1, Ordering::Relaxed);
        self.no_pts_warned.store(false, Ordering::Relaxed);
        self.worker_failure_reported.store(false, Ordering::Relaxed);
        if let Ok(mut sampler) = self.sampler.lock() {
            *sampler = Sampler::default();
        }
    }
    pub(super) fn reset_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut sampler) = self.sampler.lock() {
            *sampler = Sampler::default();
        }
    }
    fn notify_counters(&self) {
        for name in [
            "submitted-frames",
            "completed-frames",
            "failed-frames",
            "dropped-frames",
        ] {
            self.obj().notify(name);
        }
    }
    fn reap_stale_worker(&self) -> Result<(), gst::ErrorMessage> {
        let stale = self
            .state
            .lock()
            .map_err(|_error| {
                gst::error_msg!(
                    gst::ResourceError::Failed,
                    ["OCR worker state is unavailable"]
                )
            })?
            .as_ref()
            .is_some_and(|worker| worker.stop.load(Ordering::Acquire) || worker.sender.is_none());
        let worker = stale
            .then(|| self.state.lock().ok().and_then(|mut state| state.take()))
            .flatten();
        let Some(mut worker) = worker else {
            return Ok(());
        };
        if worker.thread_id == std::thread::current().id() {
            if let Ok(mut state) = self.state.lock() {
                *state = Some(worker);
            }
            return Err(gst::error_msg!(
                gst::ResourceError::Failed,
                ["OCR worker cannot restart itself"]
            ));
        }
        worker.sender.take();
        let _joined = worker.handle.join();
        Ok(())
    }
    fn input_error(
        &self,
        pts: Option<gst::ClockTime>,
        width: u32,
        height: u32,
        text: &'static str,
    ) {
        self.counters.failed.fetch_add(1, Ordering::Relaxed);
        self.obj().notify("failed-frames");
        message::post(
            &self.obj(),
            message::error_structure(
                None,
                self.generation.load(Ordering::Relaxed),
                OcrError::Input,
                text,
                pts,
                width,
                height,
            ),
        );
    }
    pub(super) fn select(&self, pts: Option<gst::ClockTime>, interval: u64) -> bool {
        let Ok(mut sampler) = self.sampler.lock() else {
            return false;
        };
        if let (Some(previous), Some(current)) = (sampler.last_seen_valid_pts, pts)
            && current < previous
        {
            self.generation.fetch_add(1, Ordering::Relaxed);
            *sampler = Sampler::default();
        }
        if let Some(pts) = pts {
            sampler.last_seen_valid_pts = Some(pts);
        }
        if interval == 0 {
            return true;
        }
        match (pts, &sampler.selection) {
            (Some(pts), Selection::Empty | Selection::SelectedWithoutPts) => {
                sampler.selection = Selection::SelectedAt(pts);
                true
            }
            (Some(pts), Selection::SelectedAt(previous))
                if pts.saturating_sub(*previous).nseconds() >= interval =>
            {
                sampler.selection = Selection::SelectedAt(pts);
                true
            }
            (None, Selection::Empty) => {
                sampler.selection = Selection::SelectedWithoutPts;
                true
            }
            _ => false,
        }
    }

    #[cfg(test)]
    pub(super) fn test_generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }
    #[cfg(test)]
    pub(super) fn test_no_pts_warned(&self) -> bool {
        self.no_pts_warned.load(Ordering::Acquire)
    }
}

impl ObjectImpl for OcrsAnalysis {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_gap_aware(true);
    }
    fn dispose(&self) {
        let _stopped = BaseTransformImpl::stop(self);
    }
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("detection-model")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("recognition-model")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("alphabet-file")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("allowed-characters")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("analysis-interval")
                    .default_value(DEFAULT_INTERVAL)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("max-model-bytes")
                    .minimum(1)
                    .maximum(MAX_MODEL_BYTES)
                    .default_value(DEFAULT_MODEL_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("max-frame-bytes")
                    .minimum(1)
                    .maximum(MAX_FRAME_BYTES)
                    .default_value(DEFAULT_FRAME_BYTES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-lines")
                    .minimum(1)
                    .maximum(512)
                    .default_value(DEFAULT_MAX_LINES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-text-length")
                    .minimum(1)
                    .maximum(1024)
                    .default_value(DEFAULT_TEXT_LENGTH)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt64::builder("submitted-frames")
                    .read_only()
                    .build(),
                glib::ParamSpecUInt64::builder("completed-frames")
                    .read_only()
                    .build(),
                glib::ParamSpecUInt64::builder("failed-frames")
                    .read_only()
                    .build(),
                glib::ParamSpecUInt64::builder("dropped-frames")
                    .read_only()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }
    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let Ok(mut settings) = self.settings.lock() else {
            return;
        };
        match pspec.name() {
            "detection-model" => {
                settings.detection_model = value
                    .get::<Option<String>>()
                    .ok()
                    .flatten()
                    .filter(|value| !value.is_empty())
                    .map(PathBuf::from);
            }
            "recognition-model" => {
                settings.recognition_model = value
                    .get::<Option<String>>()
                    .ok()
                    .flatten()
                    .filter(|value| !value.is_empty())
                    .map(PathBuf::from);
            }
            "alphabet-file" => {
                settings.alphabet_file = value
                    .get::<Option<String>>()
                    .ok()
                    .flatten()
                    .filter(|value| !value.is_empty())
                    .map(PathBuf::from);
            }
            "allowed-characters" => {
                settings.allowed_characters = value.get::<Option<String>>().ok().flatten();
            }
            "analysis-interval" => {
                if let Ok(value) = value.get() {
                    settings.analysis_interval = value;
                }
            }
            "max-model-bytes" => {
                if let Ok(value) = value.get() {
                    settings.max_model_bytes = value;
                }
            }
            "max-frame-bytes" => {
                if let Ok(value) = value.get() {
                    settings.max_frame_bytes = value;
                }
            }
            "max-lines" => {
                if let Ok(value) = value.get::<u32>() {
                    settings.max_lines = value;
                }
            }
            "max-text-length" => {
                if let Ok(value) = value.get::<u32>() {
                    settings.max_text_length = value;
                }
            }
            _ => (),
        }
    }
    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().ok();
        match pspec.name() {
            "detection-model" => settings
                .as_ref()
                .and_then(|value| value.detection_model.as_ref())
                .map(|value| value.to_string_lossy().into_owned())
                .to_value(),
            "recognition-model" => settings
                .as_ref()
                .and_then(|value| value.recognition_model.as_ref())
                .map(|value| value.to_string_lossy().into_owned())
                .to_value(),
            "alphabet-file" => settings
                .as_ref()
                .and_then(|value| value.alphabet_file.as_ref())
                .map(|value| value.to_string_lossy().into_owned())
                .to_value(),
            "allowed-characters" => settings
                .as_ref()
                .and_then(|value| value.allowed_characters.clone())
                .to_value(),
            "analysis-interval" => settings
                .as_ref()
                .map_or(DEFAULT_INTERVAL, |value| value.analysis_interval)
                .to_value(),
            "max-model-bytes" => settings
                .as_ref()
                .map_or(DEFAULT_MODEL_BYTES, |value| value.max_model_bytes)
                .to_value(),
            "max-frame-bytes" => settings
                .as_ref()
                .map_or(DEFAULT_FRAME_BYTES, |value| value.max_frame_bytes)
                .to_value(),
            "max-lines" => settings
                .as_ref()
                .map_or(DEFAULT_MAX_LINES, |value| value.max_lines)
                .to_value(),
            "max-text-length" => settings
                .as_ref()
                .map_or(DEFAULT_TEXT_LENGTH, |value| value.max_text_length)
                .to_value(),
            "submitted-frames" => self.counters.submitted.load(Ordering::Relaxed).to_value(),
            "completed-frames" => self.counters.completed.load(Ordering::Relaxed).to_value(),
            "failed-frames" => self.counters.failed.load(Ordering::Relaxed).to_value(),
            "dropped-frames" => self.counters.dropped.load(Ordering::Relaxed).to_value(),
            _ => 0_u64.to_value(),
        }
    }
}

impl GstObjectImpl for OcrsAnalysis {}

impl ElementImpl for OcrsAnalysis {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "Local OCRs Analysis",
                "Filter/Analysis/Video",
                "Recognizes text with OCRs without modifying video buffers",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });
        Some(&METADATA)
    }
    #[expect(
        clippy::expect_used,
        reason = "static RGB pad templates are infallible"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::<i32>::new(1, i32::MAX))
                .field("height", gst::IntRange::<i32>::new(1, i32::MAX))
                .build();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .expect("static OCR sink pad"),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .expect("static OCR source pad"),
            ]
        });
        TEMPLATES.as_ref()
    }

    fn change_state(
        &self,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        if matches!(
            transition,
            gst::StateChange::ReadyToPaused | gst::StateChange::PausedToPlaying
        ) && self
            .state
            .lock()
            .ok()
            .and_then(|state| state.as_ref().map(|worker| worker.thread_id))
            == Some(std::thread::current().id())
        {
            return Err(gst::StateChangeError);
        }
        self.parent_change_state(transition)
    }
}

impl BaseTransformImpl for OcrsAnalysis {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = true;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        self.reap_stale_worker()?;
        let settings = self
            .settings
            .lock()
            .map_err(|_error| {
                gst::error_msg!(
                    gst::ResourceError::Settings,
                    ["OCR settings are unavailable"]
                )
            })?
            .clone();
        #[cfg(test)]
        let test_factory = self
            .test_backend_factory
            .lock()
            .ok()
            .and_then(|slot| slot.clone());
        #[cfg(test)]
        let post_admission_hook = self
            .test_post_admission_hook
            .lock()
            .ok()
            .and_then(|slot| slot.clone());
        #[cfg(test)]
        let backend = if let Some(factory) = test_factory {
            factory().map_err(|_error| {
                gst::error_msg!(gst::ResourceError::Settings, ["invalid OCR test backend"])
            })?
        } else {
            Self::load_backend(&settings)?
        };
        #[cfg(not(test))]
        let backend = Self::load_backend(&settings)?;
        let worker = worker::start(
            backend,
            Arc::clone(&self.counters),
            self.obj().downgrade(),
            #[cfg(test)]
            post_admission_hook,
        )
        .map_err(|message| gst::error_msg!(gst::ResourceError::Failed, ["{message}"]))?;
        self.reset_run();
        *self.state.lock().map_err(|_error| {
            gst::error_msg!(
                gst::ResourceError::Failed,
                ["OCR worker state is unavailable"]
            )
        })? = Some(worker);
        self.notify_counters();
        Ok(())
    }
    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let worker = self.state.lock().ok().and_then(|mut state| state.take());
        if let Some(mut worker) = worker {
            worker.stop.store(true, Ordering::Release);
            worker.emission_open.swap(false, Ordering::AcqRel);
            worker.sender.take();
            if worker.thread_id == std::thread::current().id() {
                if let Ok(mut state) = self.state.lock() {
                    *state = Some(worker);
                }
                return Ok(());
            }
            let _joined = worker.handle.join();
        }
        if let Ok(mut sampler) = self.sampler.lock() {
            *sampler = Sampler::default();
        }
        Ok(())
    }
    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let info = gst_video::VideoInfo::from_caps(incaps)
            .map_err(|_error| gst::loggable_error!(gst::CAT_RUST, "invalid RGB video caps"))?;
        if info.format() != gst_video::VideoFormat::Rgb {
            return Err(gst::loggable_error!(
                gst::CAT_RUST,
                "ocrsanalysis requires RGB video"
            ));
        }
        *self.video_info.lock().map_err(|_error| {
            gst::loggable_error!(gst::CAT_RUST, "OCR video state is unavailable")
        })? = Some(info);
        self.parent_set_caps(incaps, outcaps)
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
    fn transform_ip_passthrough(
        &self,
        buffer: &gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self
            .settings
            .lock()
            .map_err(|_error| gst::FlowError::Error)?
            .clone();
        let info = self
            .video_info
            .lock()
            .map_err(|_error| gst::FlowError::NotNegotiated)?
            .clone()
            .ok_or(gst::FlowError::NotNegotiated)?;
        let pts = buffer.pts();
        if pts.is_none() && !self.no_pts_warned.swap(true, Ordering::AcqRel) {
            gst::warning!(
                gst::CAT_RUST,
                imp = self,
                "OCR sampling received a buffer without PTS"
            );
        }
        if !self.select(pts, settings.analysis_interval) {
            return Ok(gst::FlowSuccess::Ok);
        }
        let packed = u64::from(info.width())
            .checked_mul(u64::from(info.height()))
            .and_then(|value| value.checked_mul(3))
            .ok_or(gst::FlowError::Error)?;
        if u64::try_from(buffer.size()).map_err(|_error| gst::FlowError::Error)?
            > settings.max_frame_bytes
            || packed > settings.max_frame_bytes
        {
            self.input_error(
                pts,
                info.width(),
                info.height(),
                "OCR frame exceeds configured limit",
            );
            return Ok(gst::FlowSuccess::Ok);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let job = Job {
            buffer: buffer.clone(),
            info,
            id,
            generation: self.generation.load(Ordering::Relaxed),
            pts,
            max_frame_bytes: settings.max_frame_bytes,
            max_lines: settings.max_lines,
            max_text_length: settings.max_text_length,
            enqueued_at: Instant::now(),
        };
        let sender = self
            .state
            .lock()
            .map_err(|_error| gst::FlowError::Flushing)?
            .as_ref()
            .and_then(|worker| worker.sender.clone())
            .ok_or(gst::FlowError::Flushing)?;
        match sender.try_send(job) {
            Ok(()) => {
                self.counters.submitted.fetch_add(1, Ordering::Relaxed);
                self.obj().notify("submitted-frames");
            }
            Err(std::sync::mpsc::TrySendError::Full(_job)) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
                self.obj().notify("dropped-frames");
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_job)) => {
                if !self.worker_failure_reported.swap(true, Ordering::AcqRel) {
                    self.obj().post_error_message(gst::error_msg!(
                        gst::ResourceError::Failed,
                        ["OCR worker stopped unexpectedly"]
                    ));
                }
                return Err(gst::FlowError::Flushing);
            }
        }
        Ok(gst::FlowSuccess::Ok)
    }
}

impl OcrsAnalysis {
    fn load_backend(
        settings: &Settings,
    ) -> Result<Box<dyn backend::OcrBackend>, gst::ErrorMessage> {
        let detection = settings.detection_model.as_ref().ok_or_else(|| {
            gst::error_msg!(
                gst::ResourceError::Settings,
                ["detection-model must be set"]
            )
        })?;
        let recognition = settings.recognition_model.as_ref().ok_or_else(|| {
            gst::error_msg!(
                gst::ResourceError::Settings,
                ["recognition-model must be set"]
            )
        })?;
        backend::load(
            detection,
            recognition,
            settings.max_model_bytes,
            settings.alphabet_file.as_deref(),
            settings.allowed_characters.as_deref(),
        )
        .map_err(|_error| {
            gst::error_msg!(gst::ResourceError::Settings, ["invalid OCR model settings"])
        })
    }
}
