use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use gst::glib;
use gst::prelude::*;
use gst_video::VideoFrameExt as _;

use crate::backend::{OcrBackend, OcrError};
use crate::message;

#[cfg(test)]
pub(crate) type PostAdmissionHook = Arc<dyn Fn() + Send + Sync>;

pub(crate) struct Counters {
    pub(crate) submitted: AtomicU64,
    pub(crate) completed: AtomicU64,
    pub(crate) failed: AtomicU64,
    pub(crate) dropped: AtomicU64,
}

impl Counters {
    pub(crate) fn reset(&self) {
        self.submitted.store(0, Ordering::Relaxed);
        self.completed.store(0, Ordering::Relaxed);
        self.failed.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
    }
}

pub(crate) struct Job {
    pub(crate) buffer: gst::Buffer,
    pub(crate) info: gst_video::VideoInfo,
    pub(crate) id: u64,
    pub(crate) generation: u64,
    pub(crate) pts: Option<gst::ClockTime>,
    pub(crate) max_frame_bytes: u64,
    pub(crate) max_lines: u32,
    pub(crate) max_text_length: u32,
    pub(crate) enqueued_at: Instant,
}

pub(crate) struct Worker {
    pub(crate) sender: Option<mpsc::SyncSender<Job>>,
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) emission_open: Arc<AtomicBool>,
    pub(crate) thread_id: thread::ThreadId,
    pub(crate) handle: thread::JoinHandle<()>,
}

pub(crate) fn start(
    backend: Box<dyn OcrBackend>,
    counters: Arc<Counters>,
    weak: glib::WeakRef<crate::ocrsanalysis::OcrsAnalysis>,
    #[cfg(test)] post_admission_hook: Option<PostAdmissionHook>,
) -> Result<Worker, &'static str> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let stop = Arc::new(AtomicBool::new(false));
    let emission_open = Arc::new(AtomicBool::new(true));
    let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
    let thread_stop = Arc::clone(&stop);
    let thread_emission = Arc::clone(&emission_open);
    let handle = thread::Builder::new()
        .name("gstsmith-ocrs".to_owned())
        .spawn(move || {
            let _sent = ready_sender.send(thread::current().id());
            worker_loop(
                backend,
                receiver,
                counters,
                weak,
                thread_stop,
                thread_emission,
                #[cfg(test)]
                post_admission_hook,
            );
        })
        .map_err(|_error| "failed to create OCR worker")?;
    let thread_id = ready_receiver
        .recv()
        .map_err(|_error| "OCR worker exited before initialization")?;
    Ok(Worker {
        sender: Some(sender),
        stop,
        emission_open,
        thread_id,
        handle,
    })
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "the worker thread owns its channel and shared state for its complete lifetime"
)]
fn worker_loop(
    mut backend: Box<dyn OcrBackend>,
    receiver: mpsc::Receiver<Job>,
    counters: Arc<Counters>,
    weak: glib::WeakRef<crate::ocrsanalysis::OcrsAnalysis>,
    stop: Arc<AtomicBool>,
    emission_open: Arc<AtomicBool>,
    #[cfg(test)] post_admission_hook: Option<PostAdmissionHook>,
) {
    while let Ok(job) = receiver.recv() {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let width = job.info.width();
        let height = job.info.height();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process(&mut *backend, &job)
        }));
        let Some(element) = weak.upgrade() else {
            break;
        };
        match result {
            Ok(Ok(result)) => {
                counters.completed.fetch_add(1, Ordering::Relaxed);
                element.notify("completed-frames");
                if emission_open.load(Ordering::Acquire) {
                    #[cfg(test)]
                    if let Some(hook) = post_admission_hook.as_ref() {
                        hook();
                    }
                    let latency =
                        u64::try_from(job.enqueued_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    if let Ok(structure) = message::result_structure(
                        job.id,
                        job.generation,
                        job.pts,
                        width,
                        height,
                        latency,
                        result,
                    ) {
                        message::post(&element, structure);
                    } else {
                        post_error(
                            &element,
                            &counters,
                            &emission_open,
                            job.id,
                            job.generation,
                            job.pts,
                            width,
                            height,
                            OcrError::Inference,
                            "OCR result exceeds configured message limits",
                        );
                    }
                }
            }
            Ok(Err(error)) => post_error(
                &element,
                &counters,
                &emission_open,
                job.id,
                job.generation,
                job.pts,
                width,
                height,
                error,
                "OCR analysis failed",
            ),
            Err(_panic) => {
                counters.failed.fetch_add(1, Ordering::Relaxed);
                element.notify("failed-frames");
                if emission_open.swap(false, Ordering::AcqRel) {
                    element.post_error_message(gst::error_msg!(
                        gst::LibraryError::Failed,
                        ["OCR backend failed"]
                    ));
                }
                stop.store(true, Ordering::Release);
                break;
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the fixed public OCR error envelope requires source correlation fields"
)]
fn post_error(
    element: &crate::ocrsanalysis::OcrsAnalysis,
    counters: &Counters,
    emission_open: &AtomicBool,
    id: u64,
    generation: u64,
    pts: Option<gst::ClockTime>,
    width: u32,
    height: u32,
    error: OcrError,
    text: &'static str,
) {
    counters.failed.fetch_add(1, Ordering::Relaxed);
    element.notify("failed-frames");
    if emission_open.load(Ordering::Acquire) {
        message::post(
            element,
            message::error_structure(Some(id), generation, error, text, pts, width, height),
        );
    }
}

fn process(
    backend: &mut dyn OcrBackend,
    job: &Job,
) -> Result<crate::backend::OcrFrameResult, OcrError> {
    let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(&job.buffer, &job.info)
        .map_err(|_error| OcrError::Input)?;
    let source = frame.plane_data(0).map_err(|_error| OcrError::Input)?;
    let width = usize::try_from(job.info.width()).map_err(|_error| OcrError::Input)?;
    let height = usize::try_from(job.info.height()).map_err(|_error| OcrError::Input)?;
    let row_bytes = width.checked_mul(3).ok_or(OcrError::Input)?;
    let packed_len = row_bytes.checked_mul(height).ok_or(OcrError::Input)?;
    if u64::try_from(packed_len).map_err(|_error| OcrError::Input)? > job.max_frame_bytes {
        return Err(OcrError::Input);
    }
    let stride = frame
        .plane_stride()
        .first()
        .copied()
        .ok_or(OcrError::Input)?;
    let stride = usize::try_from(stride).map_err(|_error| OcrError::Input)?;
    if stride < row_bytes {
        return Err(OcrError::Input);
    }
    if stride == row_bytes && source.len() >= packed_len {
        return backend.recognize(
            source.get(..packed_len).ok_or(OcrError::Input)?,
            job.info.width(),
            job.info.height(),
            job.max_lines,
            job.max_text_length,
        );
    }
    let mut packed = Vec::with_capacity(packed_len);
    for row in 0..height {
        let start = row.checked_mul(stride).ok_or(OcrError::Input)?;
        let end = start.checked_add(row_bytes).ok_or(OcrError::Input)?;
        let bytes = source.get(start..end).ok_or(OcrError::Input)?;
        packed.extend_from_slice(bytes);
    }
    backend.recognize(
        &packed,
        job.info.width(),
        job.info.height(),
        job.max_lines,
        job.max_text_length,
    )
}
