use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, Once, mpsc};
use std::time::{Duration, Instant};

use gst::glib;
use gst::glib::subclass::types::ObjectSubclassIsExt;
use gst::prelude::*;
use gst_base::subclass::prelude::BaseTransformImpl;

use crate::backend::{OcrBackend, OcrError, OcrFrameResult};

struct EmptyBackend;

impl OcrBackend for EmptyBackend {
    fn recognize(
        &mut self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        Ok(OcrFrameResult { lines: Vec::new() })
    }
}

struct BlockingBackend {
    started: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

struct RecoveringBackend {
    outcomes: VecDeque<Result<OcrFrameResult, OcrError>>,
    observed: mpsc::Sender<bool>,
}

struct PanicBackend {
    started: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

struct OldCapsBackend {
    calls: u8,
    started: mpsc::Sender<()>,
    observed: mpsc::Sender<(u32, u32)>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

struct PixelCaptureBackend {
    captured: mpsc::Sender<Vec<u8>>,
}

struct CorrelationBackend {
    calls: u8,
    started: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

impl OcrBackend for CorrelationBackend {
    fn recognize(
        &mut self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        self.calls = self.calls.checked_add(1).ok_or(OcrError::Inference)?;
        if self.calls == 1 {
            self.started
                .send(())
                .map_err(|_error| OcrError::Inference)?;
            let (released, wake) = &*self.release;
            let guard = released.lock().map_err(|_error| OcrError::Inference)?;
            let _guard = wake
                .wait_while(guard, |released| !*released)
                .map_err(|_error| OcrError::Inference)?;
        }
        Ok(OcrFrameResult { lines: Vec::new() })
    }
}

impl OcrBackend for PixelCaptureBackend {
    fn recognize(
        &mut self,
        rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        self.captured
            .send(rgb.to_vec())
            .map_err(|_error| OcrError::Inference)?;
        Ok(OcrFrameResult { lines: Vec::new() })
    }
}

impl OcrBackend for OldCapsBackend {
    fn recognize(
        &mut self,
        _rgb: &[u8],
        width: u32,
        height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        self.calls = self.calls.checked_add(1).ok_or(OcrError::Inference)?;
        if self.calls == 1 {
            self.started
                .send(())
                .map_err(|_error| OcrError::Inference)?;
            let (released, wake) = &*self.release;
            let guard = released.lock().map_err(|_error| OcrError::Inference)?;
            let _guard = wake
                .wait_while(guard, |released| !*released)
                .map_err(|_error| OcrError::Inference)?;
        } else {
            self.observed
                .send((width, height))
                .map_err(|_error| OcrError::Inference)?;
        }
        Ok(OcrFrameResult { lines: Vec::new() })
    }
}

impl OcrBackend for PanicBackend {
    #[expect(
        clippy::panic_in_result_fn,
        reason = "this fake backend deliberately models a dependency panic at the worker boundary"
    )]
    fn recognize(
        &mut self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        self.started
            .send(())
            .map_err(|_error| OcrError::Inference)?;
        let (released, wake) = &*self.release;
        let guard = released.lock().map_err(|_error| OcrError::Inference)?;
        let _guard = wake
            .wait_while(guard, |released| !*released)
            .map_err(|_error| OcrError::Inference)?;
        panic!("test-only dependency panic");
    }
}

impl OcrBackend for RecoveringBackend {
    fn recognize(
        &mut self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        let outcome = self.outcomes.pop_front().ok_or(OcrError::Inference)?;
        self.observed
            .send(outcome.is_ok())
            .map_err(|_error| OcrError::Inference)?;
        outcome
    }
}

impl OcrBackend for BlockingBackend {
    fn recognize(
        &mut self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _max_lines: u32,
        _max_text_length: u32,
    ) -> Result<OcrFrameResult, OcrError> {
        self.started
            .send(())
            .map_err(|_error| OcrError::Inference)?;
        let (released, wake) = &*self.release;
        let guard = released.lock().map_err(|_error| OcrError::Inference)?;
        let _guard = wake
            .wait_while(guard, |released| !*released)
            .map_err(|_error| OcrError::Inference)?;
        Ok(OcrFrameResult { lines: Vec::new() })
    }
}

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().expect("initializing GStreamer for OCR tests");
    });
}

fn fake_element() -> super::OcrsAnalysis {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element
        .imp()
        .set_test_backend_factory(Arc::new(|| Ok(Box::new(EmptyBackend))));
    element
}

fn rgb_caps() -> gst::Caps {
    rgb_caps_with_width(1)
}

fn rgb_caps_with_width(width: i32) -> gst::Caps {
    rgb_caps_with_size(width, 1)
}

fn rgb_caps_with_size(width: i32, height: i32) -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", width)
        .field("height", height)
        .build()
}

fn rgb_buffer(pts: u64) -> gst::Buffer {
    let mut buffer = gst::Buffer::with_size(4).expect("allocating RGB buffer with aligned stride");
    buffer
        .get_mut()
        .expect("new RGB buffer is writable")
        .set_pts(gst::ClockTime::from_nseconds(pts));
    buffer
}

fn wait_until(timeout: Duration, predicate: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::yield_now();
    }
    predicate()
}

#[test]
fn properties_have_documented_defaults() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    assert_eq!(element.property::<u64>("analysis-interval"), 500_000_000);
    assert_eq!(element.property::<u32>("max-lines"), 128);
}

#[test]
fn sampling_interval_and_no_pts_rules_are_deterministic() {
    let analysis = super::imp::OcrsAnalysis::default();
    assert!(analysis.select(Some(gst::ClockTime::from_nseconds(0)), 10));
    assert!(!analysis.select(Some(gst::ClockTime::from_nseconds(9)), 10));
    assert!(analysis.select(Some(gst::ClockTime::from_nseconds(10)), 10));
    let no_pts = super::imp::OcrsAnalysis::default();
    assert!(no_pts.select(None, 10));
    assert!(!no_pts.select(None, 10));
    assert!(no_pts.select(None, 0));
}

#[test]
fn sampling_no_pts_warning_state_is_once_per_run_and_resets() {
    let element = fake_element();
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting warning-state worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    let buffer = gst::Buffer::with_size(4).expect("allocating no-PTS buffer");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &buffer)
        .expect("first no-PTS frame");
    assert!(element.imp().test_no_pts_warned());
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &buffer)
        .expect("second no-PTS frame");
    assert!(element.imp().test_no_pts_warned());
    BaseTransformImpl::stop(element.imp()).expect("stopping warning-state worker");
    BaseTransformImpl::start(element.imp()).expect("restarting warning-state worker");
    assert!(!element.imp().test_no_pts_warned());
    BaseTransformImpl::stop(element.imp()).expect("stopping restarted worker");
}

#[test]
fn generation_backward_pts_resets_selection() {
    let analysis = super::imp::OcrsAnalysis::default();
    assert!(analysis.select(Some(gst::ClockTime::from_nseconds(100)), 100));
    assert!(!analysis.select(Some(gst::ClockTime::from_nseconds(150)), 100));
    assert!(analysis.select(Some(gst::ClockTime::from_nseconds(50)), 100));
    assert_eq!(analysis.test_generation(), 2);
}

#[test]
fn generation_events_advance_the_captured_message_generation() {
    let element = fake_element();
    let mut harness = gst_check::Harness::with_element(&element, Some("sink"), Some("src"));
    harness.set_src_caps(rgb_caps());
    harness.play();
    let before = element.imp().test_generation();
    assert!(harness.push_event(gst::event::StreamStart::new("ocr-test")));
    assert_eq!(element.imp().test_generation(), before + 1);
    assert!(
        harness.push_event(gst::event::Segment::new(&gst::FormattedSegment::<
            gst::ClockTime,
        >::new()))
    );
    assert_eq!(element.imp().test_generation(), before + 2);
    assert!(harness.push_event(gst::event::FlushStop::new(false)));
    assert_eq!(element.imp().test_generation(), before + 3);
}

#[test]
fn worker_stale_inflight_results_keep_captured_source_correlation() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (started_sender, started_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = Arc::new(Mutex::new(Some(Box::new(CorrelationBackend {
        calls: 0,
        started: started_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting correlation worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(10))
        .expect("submitting old-generation frame");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("blocking first job");
    element.imp().reset_generation();
    let new_caps = rgb_caps_with_width(2);
    BaseTransformImpl::set_caps(element.imp(), &new_caps, &new_caps)
        .expect("renegotiating source size");
    let mut new_buffer = gst::Buffer::with_size(8).expect("allocating new-size RGB buffer");
    new_buffer
        .get_mut()
        .expect("new buffer writable")
        .set_pts(gst::ClockTime::from_nseconds(20));
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &new_buffer)
        .expect("queueing new-generation frame");
    let (released, wake) = &*release;
    *released.lock().expect("releasing old job") = true;
    wake.notify_all();
    let first = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Element])
        .expect("old result");
    let second = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Element])
        .expect("new result");
    let structure = |message: gst::Message| {
        let gst::MessageView::Element(element) = message.view() else {
            panic!("expected OCR element result message");
        };
        element.structure().expect("result structure").to_owned()
    };
    let first = structure(first);
    let second = structure(second);
    assert_eq!(first.get::<u64>("request-id"), Ok(1));
    assert_eq!(first.get::<u64>("generation"), Ok(1));
    assert_eq!(
        first.get::<gst::ClockTime>("source-pts"),
        Ok(gst::ClockTime::from_nseconds(10))
    );
    assert_eq!(first.get::<u32>("source-width"), Ok(1));
    assert_eq!(first.get::<u32>("source-height"), Ok(1));
    assert_eq!(second.get::<u64>("request-id"), Ok(2));
    assert_eq!(second.get::<u64>("generation"), Ok(2));
    assert_eq!(
        second.get::<gst::ClockTime>("source-pts"),
        Ok(gst::ClockTime::from_nseconds(20))
    );
    assert_eq!(second.get::<u32>("source-width"), Ok(2));
    assert_eq!(second.get::<u32>("source-height"), Ok(1));
    BaseTransformImpl::stop(element.imp()).expect("stopping correlation worker");
}

#[test]
fn properties_test_factory_bypasses_model_paths_at_start() {
    let element = fake_element();
    BaseTransformImpl::start(element.imp()).expect("starting fake OCR backend");
    BaseTransformImpl::stop(element.imp()).expect("stopping fake OCR backend");
}

#[test]
fn input_size_rejection_posts_correlated_sanitized_error_without_request_id() {
    let element = fake_element();
    element.set_property("max-frame-bytes", 1_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting input-limit worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(42))
        .expect("passing through rejected frame");
    let message = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Element])
        .expect("receiving input error message");
    let gst::MessageView::Element(element_message) = message.view() else {
        panic!("expected input error element message");
    };
    let structure = element_message.structure().expect("input error structure");
    assert_eq!(structure.name(), "ocr-error");
    assert_eq!(
        structure.get::<String>("kind").ok().as_deref(),
        Some("input")
    );
    assert!(!structure.has_field("request-id"));
    assert_eq!(structure.get::<u64>("generation"), Ok(1));
    assert_eq!(
        structure.get::<gst::ClockTime>("source-pts"),
        Ok(gst::ClockTime::from_nseconds(42))
    );
    assert_eq!(structure.get::<u32>("source-width"), Ok(1));
    assert_eq!(structure.get::<u32>("source-height"), Ok(1));
    assert_eq!(element.property::<u64>("failed-frames"), 1);
    assert_eq!(element.property::<u64>("submitted-frames"), 0);
    BaseTransformImpl::stop(element.imp()).expect("stopping input-limit worker");
}

#[test]
fn passthrough_rgb_buffer_keeps_identity_timing_and_gap_flag() {
    let element = fake_element();
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting fake OCR backend");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    let mut buffer = gst::Buffer::with_size(3).expect("allocating RGB buffer");
    let buffer_ref = buffer.get_mut().expect("new buffer is writable");
    buffer_ref.set_pts(gst::ClockTime::from_nseconds(42));
    buffer_ref.set_flags(gst::BufferFlags::GAP);
    let pointer = buffer.as_ptr();
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &buffer)
        .expect("passing through RGB buffer");
    assert_eq!(buffer.as_ptr(), pointer);
    assert_eq!(buffer.pts(), Some(gst::ClockTime::from_nseconds(42)));
    assert!(buffer.flags().contains(gst::BufferFlags::GAP));
    BaseTransformImpl::stop(element.imp()).expect("stopping fake OCR backend");
}

#[test]
fn queue_full_drops_newest_frame_without_blocking_the_streaming_callback() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let (started_sender, started_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = Arc::new(Mutex::new(Some(Box::new(BlockingBackend {
        started: started_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting blocking worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting active job");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("worker starts first job");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(1))
        .expect("queueing second job");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(2))
        .expect("dropping newest job");
    assert_eq!(element.property::<u64>("dropped-frames"), 1);
    let (released, wake) = &*release;
    *released.lock().expect("unlocking worker") = true;
    wake.notify_all();
    BaseTransformImpl::stop(element.imp()).expect("joining released worker");
}

#[test]
fn worker_shutdown_and_recovery_allow_a_fresh_fake_run() {
    let element = fake_element();
    BaseTransformImpl::start(element.imp()).expect("starting first fake worker");
    BaseTransformImpl::stop(element.imp()).expect("joining first fake worker");
    BaseTransformImpl::start(element.imp()).expect("starting replacement fake worker");
    BaseTransformImpl::stop(element.imp()).expect("joining replacement fake worker");
}

#[test]
fn shutdown_dispose_reaps_the_worker_without_detaching_it() {
    let element = fake_element();
    BaseTransformImpl::start(element.imp()).expect("starting disposable fake worker");
    drop(element);
}

#[test]
fn recovery_reaps_a_closed_stale_handle_before_restart() {
    let element = fake_element();
    BaseTransformImpl::start(element.imp()).expect("starting fake worker");
    {
        let mut state = element.imp().state.lock().expect("locking worker state");
        let worker = state.as_mut().expect("stored worker");
        worker
            .stop
            .store(true, std::sync::atomic::Ordering::Release);
        worker
            .emission_open
            .store(false, std::sync::atomic::Ordering::Release);
        worker.sender.take();
    }
    BaseTransformImpl::start(element.imp()).expect("reaping stale worker before restart");
    BaseTransformImpl::stop(element.imp()).expect("stopping replacement worker");
}

#[test]
fn worker_ordinary_failure_recovers_for_the_next_selected_job() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let (observed_sender, observed_receiver) = mpsc::channel();
    let backend = Arc::new(Mutex::new(Some(Box::new(RecoveringBackend {
        outcomes: VecDeque::from([
            Err(OcrError::Inference),
            Ok(OcrFrameResult { lines: Vec::new() }),
        ]),
        observed: observed_sender,
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting recovering worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting recoverable failure");
    assert!(
        !observed_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("first backend call")
    );
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(1))
        .expect("submitting recovery job");
    assert!(
        observed_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("second backend call")
    );
    assert!(wait_until(Duration::from_secs(1), || {
        element.property::<u64>("completed-frames") == 1
    }));
    assert_eq!(element.property::<u64>("failed-frames"), 1);
    assert_eq!(element.property::<u64>("completed-frames"), 1);
    BaseTransformImpl::stop(element.imp()).expect("stopping recovered worker");
}

#[test]
fn worker_dependency_panic_is_contained_and_posts_one_fatal_error() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (started_sender, started_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = Arc::new(Mutex::new(Some(Box::new(PanicBackend {
        started: started_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting panic worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting panic job");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("worker starts panic job");
    let (released, wake) = &*release;
    *released.lock().expect("releasing panic backend") = true;
    wake.notify_all();
    let message = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Error])
        .expect("receiving one fatal worker error");
    assert!(matches!(message.view(), gst::MessageView::Error(_)));
    assert_eq!(element.property::<u64>("failed-frames"), 1);
    assert!(
        bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Error])
            .is_none()
    );
    assert!(
        bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Element])
            .is_none()
    );
    assert!(wait_until(Duration::from_secs(1), || {
        element
            .imp()
            .state
            .lock()
            .expect("locking failed worker state")
            .as_ref()
            .is_some_and(|worker| worker.handle.is_finished())
    }));
    assert_eq!(
        BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(1)),
        Err(gst::FlowError::Flushing)
    );
    assert_eq!(
        BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(2)),
        Err(gst::FlowError::Flushing)
    );
    let resource_error = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Error])
        .expect("receiving one disconnected-worker resource error");
    assert!(matches!(resource_error.view(), gst::MessageView::Error(_)));
    assert!(
        bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Error])
            .is_none()
    );
    BaseTransformImpl::stop(element.imp()).expect("reaping panic worker");
}

#[test]
fn shutdown_closes_admission_before_active_work_can_post() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (started_sender, started_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = Arc::new(Mutex::new(Some(Box::new(BlockingBackend {
        started: started_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting blocking worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting active job");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("worker starts active job");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(1))
        .expect("queueing work to discard during stop");
    let admission = element
        .imp()
        .state
        .lock()
        .expect("locking active worker")
        .as_ref()
        .expect("stored active worker")
        .emission_open
        .clone();
    let stopping_element = element.clone();
    let stopping = std::thread::spawn(move || BaseTransformImpl::stop(stopping_element.imp()));
    assert!(wait_until(Duration::from_secs(1), || !admission
        .load(std::sync::atomic::Ordering::Acquire)));
    let (released, wake) = &*release;
    *released.lock().expect("releasing active worker") = true;
    wake.notify_all();
    stopping
        .join()
        .expect("joining stop caller")
        .expect("stopping after active work");
    assert_eq!(element.property::<u64>("submitted-frames"), 2);
    assert_eq!(element.property::<u64>("completed-frames"), 1);
    assert!(
        bus.timed_pop_filtered(
            gst::ClockTime::ZERO,
            &[gst::MessageType::Element, gst::MessageType::Error],
        )
        .is_none()
    );
}

#[test]
fn shutdown_permits_one_post_that_won_admission_before_close() {
    init();
    let element = fake_element();
    element.set_property("analysis-interval", 0_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (admitted_sender, admitted_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    element.imp().set_test_post_admission_hook(Some(Arc::new({
        let release = Arc::clone(&release);
        move || {
            let _sent = admitted_sender.send(());
            let (released, wake) = &*release;
            let guard = released.lock().expect("locking admission barrier");
            let _guard = wake
                .wait_while(guard, |released| !*released)
                .expect("waiting for admission release");
        }
    })));
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting admitted worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting admitted result");
    admitted_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("worker wins admission before stop");
    let stopping_element = element.clone();
    let stopping = std::thread::spawn(move || BaseTransformImpl::stop(stopping_element.imp()));
    let (released, wake) = &*release;
    *released.lock().expect("releasing admitted post") = true;
    wake.notify_all();
    let message = bus
        .timed_pop_filtered(gst::ClockTime::SECOND, &[gst::MessageType::Element])
        .expect("receiving permitted admitted result");
    assert!(matches!(message.view(), gst::MessageView::Element(_)));
    stopping
        .join()
        .expect("joining stop caller")
        .expect("stopping admitted worker");
    assert!(
        bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Element])
            .is_none()
    );
    element.imp().set_test_post_admission_hook(None);
}

#[test]
fn worker_reentrant_stop_rejects_upward_change_and_external_restart_reaps_handle() {
    init();
    let element = fake_element();
    element.set_property("analysis-interval", 0_u64);
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (handled_sender, handled_receiver) = mpsc::channel();
    let handled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    bus.set_sync_handler({
        let element = element.clone();
        let handled = Arc::clone(&handled);
        move |_bus, message| {
            if message.type_() == gst::MessageType::Element
                && !handled.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                let stop_ok = BaseTransformImpl::stop(element.imp()).is_ok();
                let upward_rejected = matches!(
                    gst::subclass::prelude::ElementImpl::change_state(
                        element.imp(),
                        gst::StateChange::PausedToPlaying,
                    ),
                    Err(gst::StateChangeError)
                );
                let _sent = handled_sender.send((stop_ok, upward_rejected));
            }
            gst::BusSyncReply::Drop
        }
    });
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting reentrant worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting reentrant result");
    assert_eq!(
        handled_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("handling worker result on bus sync handler"),
        (true, true)
    );
    BaseTransformImpl::stop(element.imp()).expect("externally reaping reentrant worker");
    BaseTransformImpl::start(element.imp()).expect("restarting after stale handle reap");
    BaseTransformImpl::stop(element.imp()).expect("stopping replacement worker");
    bus.unset_sync_handler();
}

#[test]
fn worker_reentrant_error_handler_stops_without_recursive_diagnostic() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let (started_sender, started_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let first = Arc::new(Mutex::new(Some(Box::new(PanicBackend {
        started: started_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    let started_once = Arc::new(std::sync::atomic::AtomicBool::new(false));
    element.imp().set_test_backend_factory(Arc::new({
        let first = Arc::clone(&first);
        let started_once = Arc::clone(&started_once);
        move || {
            if started_once.swap(true, std::sync::atomic::Ordering::AcqRel) {
                Ok(Box::new(EmptyBackend) as Box<dyn OcrBackend>)
            } else {
                first
                    .lock()
                    .map_err(|_error| OcrError::Inference)?
                    .take()
                    .ok_or(OcrError::Inference)
            }
        }
    }));
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let (handled_sender, handled_receiver) = mpsc::channel();
    let handled = Arc::new(std::sync::atomic::AtomicU64::new(0));
    bus.set_sync_handler({
        let element = element.clone();
        let handled = Arc::clone(&handled);
        move |_bus, message| {
            if message.type_() == gst::MessageType::Error {
                let count = handled.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
                let stop_ok = BaseTransformImpl::stop(element.imp()).is_ok();
                let upward_rejected = matches!(
                    gst::subclass::prelude::ElementImpl::change_state(
                        element.imp(),
                        gst::StateChange::PausedToPlaying,
                    ),
                    Err(gst::StateChangeError)
                );
                let _sent = handled_sender.send((count, stop_ok, upward_rejected));
            }
            gst::BusSyncReply::Drop
        }
    });
    let caps = rgb_caps();
    BaseTransformImpl::start(element.imp()).expect("starting reentrant panic worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting panic job");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("blocking panic worker");
    let (released, wake) = &*release;
    *released.lock().expect("releasing panic backend") = true;
    wake.notify_all();
    assert_eq!(
        handled_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("handling fatal error"),
        (1, true, true)
    );
    assert_eq!(handled.load(std::sync::atomic::Ordering::Acquire), 1);
    BaseTransformImpl::stop(element.imp()).expect("externally reaping worker handle");
    BaseTransformImpl::start(element.imp()).expect("restarting after reentrant error");
    BaseTransformImpl::stop(element.imp()).expect("stopping replacement worker");
    bus.unset_sync_handler();
}

#[test]
fn worker_uses_queued_job_video_info_after_caps_renegotiation() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let (started_sender, started_receiver) = mpsc::channel();
    let (observed_sender, observed_receiver) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = Arc::new(Mutex::new(Some(Box::new(OldCapsBackend {
        calls: 0,
        started: started_sender,
        observed: observed_sender,
        release: Arc::clone(&release),
    }) as Box<dyn OcrBackend>)));
    element.imp().set_test_backend_factory(Arc::new(move || {
        backend
            .lock()
            .map_err(|_error| OcrError::Inference)?
            .take()
            .ok_or(OcrError::Inference)
    }));
    let old_caps = rgb_caps();
    let new_caps = rgb_caps_with_width(2);
    BaseTransformImpl::start(element.imp()).expect("starting old-caps worker");
    BaseTransformImpl::set_caps(element.imp(), &old_caps, &old_caps).expect("setting old caps");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(0))
        .expect("submitting active old-caps job");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("worker starts active job");
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &rgb_buffer(1))
        .expect("queueing old-caps job");
    BaseTransformImpl::set_caps(element.imp(), &new_caps, &new_caps)
        .expect("renegotiating to new caps");
    let (released, wake) = &*release;
    *released.lock().expect("releasing active job") = true;
    wake.notify_all();
    assert_eq!(
        observed_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("processing queued old-caps job"),
        (1, 1)
    );
    BaseTransformImpl::stop(element.imp()).expect("stopping old-caps worker");
}

#[test]
fn worker_packs_padded_video_meta_rows_and_preserves_passthrough_meta() {
    init();
    let element: super::OcrsAnalysis = glib::Object::new();
    element.set_property("analysis-interval", 0_u64);
    let (captured_sender, captured_receiver) = mpsc::channel();
    element.imp().set_test_backend_factory(Arc::new(move || {
        Ok(Box::new(PixelCaptureBackend {
            captured: captured_sender.clone(),
        }))
    }));
    let caps = rgb_caps_with_size(1, 2);
    BaseTransformImpl::start(element.imp()).expect("starting video-meta worker");
    BaseTransformImpl::set_caps(element.imp(), &caps, &caps).expect("setting RGB caps");
    let mut buffer = gst::Buffer::with_size(10).expect("allocating padded RGB buffer");
    {
        let mut map = buffer
            .get_mut()
            .expect("new buffer is writable")
            .map_writable()
            .expect("mapping padded RGB fixture");
        map.as_mut_slice()
            .copy_from_slice(&[99, 99, 1, 2, 3, 88, 88, 4, 5, 6]);
    }
    {
        let buffer_ref = buffer.get_mut().expect("new buffer is writable");
        gst_video::VideoMeta::add_full(
            buffer_ref,
            gst_video::VideoFrameFlags::empty(),
            gst_video::VideoFormat::Rgb,
            1,
            2,
            &[2],
            &[5],
        )
        .expect("adding padded per-buffer video metadata");
        buffer_ref.set_pts(gst::ClockTime::from_nseconds(11));
        buffer_ref.set_dts(gst::ClockTime::from_nseconds(7));
        buffer_ref.set_duration(gst::ClockTime::from_nseconds(5));
        buffer_ref.set_offset(3);
        buffer_ref.set_offset_end(13);
        buffer_ref.set_flags(gst::BufferFlags::GAP);
    }
    let shared = buffer.clone();
    let pointer = buffer.as_ptr();
    BaseTransformImpl::transform_ip_passthrough(element.imp(), &buffer)
        .expect("passing through padded video-meta buffer");
    assert_eq!(buffer.as_ptr(), pointer);
    assert_eq!(shared.as_ptr(), pointer);
    assert_eq!(buffer.pts(), Some(gst::ClockTime::from_nseconds(11)));
    assert_eq!(buffer.dts(), Some(gst::ClockTime::from_nseconds(7)));
    assert_eq!(buffer.duration(), Some(gst::ClockTime::from_nseconds(5)));
    assert_eq!(buffer.offset(), 3);
    assert_eq!(buffer.offset_end(), 13);
    assert!(buffer.flags().contains(gst::BufferFlags::GAP));
    assert_eq!(
        shared
            .map_readable()
            .expect("reading shared passthrough buffer")
            .as_slice(),
        &[99, 99, 1, 2, 3, 88, 88, 4, 5, 6]
    );
    let meta = buffer
        .meta::<gst_video::VideoMeta>()
        .expect("preserving video metadata on passthrough");
    assert_eq!(meta.offset(), &[2]);
    assert_eq!(meta.stride(), &[5]);
    assert!(
        buffer
            .meta::<gst_video::VideoRegionOfInterestMeta>()
            .is_none()
    );
    assert_eq!(
        captured_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("packing padded rows for the worker"),
        vec![1, 2, 3, 4, 5, 6]
    );
    BaseTransformImpl::stop(element.imp()).expect("stopping video-meta worker");
}
