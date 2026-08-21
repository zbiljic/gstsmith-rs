use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, LazyLock};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use parking_lot::Mutex;
use regex::Regex;

use crate::metrics::{Metrics, MetricsSlot};
use crate::server::{self, ServerHandle};

const DEFAULT_LISTEN: &str = "127.0.0.1:9099";

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "prometheus",
        gst::DebugColorFlags::empty(),
        Some("Prometheus metrics tracer"),
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
    listen: String,
    include_filter: Option<String>,
    exclude_filter: Option<String>,
    max_pad_series: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            listen: DEFAULT_LISTEN.to_owned(),
            include_filter: None,
            exclude_filter: None,
            max_pad_series: 256,
        }
    }
}

#[derive(Default)]
struct RuntimeState {
    bound_address: String,
    server: Option<ServerHandle>,
}

#[derive(Default)]
pub struct PrometheusTracer {
    settings: Mutex<Settings>,
    // gstreamer-rs applies GST_TRACERS parameters with ordinary set_property() calls from its
    // constructed() wrapper, which rejects CONSTRUCT_ONLY properties. The properties below use
    // CONSTRUCT for compatibility, and this flag preserves their startup-only contract. This can
    // become CONSTRUCT_ONLY again if upstream applies tracer parameters as construction values.
    settings_frozen: std::sync::atomic::AtomicBool,
    runtime: Mutex<RuntimeState>,
    metrics: MetricsSlot,
}

#[glib::object_subclass]
impl ObjectSubclass for PrometheusTracer {
    const NAME: &'static str = "GstSmithPrometheusTracer";
    type Type = super::PrometheusTracer;
    type ParentType = gst::Tracer;
}

impl ObjectImpl for PrometheusTracer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("listen")
                    .nick("Listen address")
                    .blurb("Numeric socket address for the OpenMetrics endpoint")
                    .default_value(Some(DEFAULT_LISTEN))
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("include-filter")
                    .nick("Include filter")
                    .blurb("Optional regular expression selecting metric scopes")
                    .construct()
                    .build(),
                glib::ParamSpecString::builder("exclude-filter")
                    .nick("Exclude filter")
                    .blurb("Optional regular expression excluding metric scopes")
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
                glib::ParamSpecString::builder("bound-address")
                    .nick("Bound address")
                    .blurb("Actual bound HTTP listener address")
                    .read_only()
                    .build(),
                glib::ParamSpecBoolean::builder("server-running")
                    .nick("Server running")
                    .blurb("Whether the metrics server started successfully")
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
                "Ignoring change to startup-only property {}",
                pspec.name()
            );
            return;
        }
        let mut settings = self.settings.lock();
        match pspec.name() {
            "listen" => {
                if let Ok(value) = value.get::<String>() {
                    settings.listen = value;
                }
            }
            "include-filter" => {
                if let Ok(value) = value.get::<Option<String>>() {
                    settings.include_filter = value;
                }
            }
            "exclude-filter" => {
                if let Ok(value) = value.get::<Option<String>>() {
                    settings.exclude_filter = value;
                }
            }
            "max-pad-series" => {
                if let Ok(value) = value.get::<u32>() {
                    settings.max_pad_series = value;
                }
            }
            _ => gst::warning!(CAT, imp = self, "Unknown property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "listen" => self.settings.lock().listen.to_value(),
            "include-filter" => self.settings.lock().include_filter.to_value(),
            "exclude-filter" => self.settings.lock().exclude_filter.to_value(),
            "max-pad-series" => self.settings.lock().max_pad_series.to_value(),
            "bound-address" => self.runtime.lock().bound_address.to_value(),
            "server-running" => self
                .runtime
                .lock()
                .server
                .as_ref()
                .is_some_and(ServerHandle::is_running)
                .to_value(),
            _ => pspec.default_value().clone(),
        }
    }

    fn constructed(&self) {
        self.settings_frozen
            .store(true, std::sync::atomic::Ordering::Release);
        self.parent_constructed();

        let settings = self.settings.lock().clone();
        let address = match settings.listen.parse::<SocketAddr>() {
            Ok(address) => address,
            Err(error) => {
                gst::error!(CAT, imp = self, "Invalid numeric listen address: {error}");
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
        let listener = match TcpListener::bind(address) {
            Ok(listener) => listener,
            Err(error) => {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to bind Prometheus endpoint: {error}"
                );
                return;
            }
        };
        let metrics = Metrics::new(include, exclude, settings.max_pad_series as usize);
        let server = match server::start(listener, Arc::clone(&metrics)) {
            Ok(server) => server,
            Err(error) => {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to start Prometheus endpoint: {error}"
                );
                return;
            }
        };
        if !self.metrics.install(metrics) {
            let mut server = server;
            if let Err(error) = server.stop() {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to stop duplicate Prometheus endpoint: {error}"
                );
            }
            gst::error!(
                CAT,
                imp = self,
                "Prometheus metrics were already initialized"
            );
            return;
        }
        {
            let mut runtime = self.runtime.lock();
            runtime.bound_address = server.address.to_string();
            runtime.server = Some(server);
        }
        self.obj().notify("bound-address");
        self.obj().notify("server-running");

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
        let mut server = self.runtime.lock().server.take();
        if let Some(server) = server.as_mut()
            && let Err(error) = server.stop()
        {
            gst::error!(
                CAT,
                imp = self,
                "Failed to stop Prometheus endpoint: {error}"
            );
        }
        self.obj().notify("server-running");
    }
}

impl GstObjectImpl for PrometheusTracer {}

impl TracerImpl for PrometheusTracer {
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
        let bytes = u64::try_from(buffer.size()).unwrap_or(u64::MAX);
        self.metrics.record_push(pad, 1, bytes);
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

impl PrometheusTracer {
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

fn compile_filter(name: &str, value: Option<&str>) -> Result<Option<Regex>, String> {
    value
        .map(|value| Regex::new(value).map_err(|error| format!("Invalid {name}: {error}")))
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::time::Duration;

    use super::*;

    static TRACER_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct HandoffRelease(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);

    impl HandoffRelease {
        fn release(&self) {
            let (released, condition) = &*self.0;
            let mut released = released
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *released = true;
            condition.notify_all();
        }
    }

    impl Drop for HandoffRelease {
        fn drop(&mut self) {
            self.release();
        }
    }

    fn scrape(address: &str) -> String {
        let mut stream = std::net::TcpStream::connect(address).expect("connecting to endpoint");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("setting read timeout");
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("writing scrape request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("reading scrape response");
        response
    }

    fn response_body(response: &str) -> &str {
        response
            .split_once("\r\n\r\n")
            .map_or(response, |(_headers, body)| body)
    }

    fn assert_sample(metrics: &str, sample: &str, value: &str) {
        let expected = format!("{sample} {value}");
        assert!(
            metrics.lines().any(|line| line == expected),
            "missing {expected:?} in:\n{metrics}"
        );
    }

    fn wait_for_eos(pipeline: &gst::Pipeline) {
        let bus = pipeline.bus().expect("pipeline bus");
        let message = bus
            .timed_pop_filtered(
                gst::ClockTime::from_seconds(5),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            )
            .expect("pipeline completion message");
        assert_eq!(message.type_(), gst::MessageType::Eos);
    }

    fn finite_queue_pipeline(
        name: &str,
        source_name: &str,
        queue_name: &str,
    ) -> (gst::Pipeline, gst::Element, gst::Element) {
        let pipeline = gst::Pipeline::builder().name(name).build();
        let source = gst::ElementFactory::make("fakesrc")
            .name(source_name)
            .property("num-buffers", 1_i32)
            .build()
            .expect("constructing finite source");
        let queue = gst::ElementFactory::make("queue")
            .name(queue_name)
            .build()
            .expect("constructing queue");
        let sink = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .build()
            .expect("constructing sink");
        pipeline
            .add_many([&source, &queue, &sink])
            .expect("adding finite pipeline elements");
        gst::Element::link_many([&source, &queue, &sink]).expect("linking finite pipeline");
        (pipeline, source, queue)
    }

    fn blocked_queue_pipeline() -> (
        gst::Pipeline,
        gst::Element,
        gst::Element,
        gst::Element,
        HandoffRelease,
        std::sync::mpsc::Receiver<()>,
    ) {
        let pipeline = gst::Pipeline::builder().name("queue_pipeline").build();
        let source = gst::ElementFactory::make("fakesrc")
            .property("num-buffers", 4_i32)
            .property_from_str("sizetype", "fixed")
            .property("sizemax", 100_i32)
            .property("datarate", 100_i32)
            .build()
            .expect("constructing queue source");
        let queue = gst::ElementFactory::make("queue")
            .name("observed_queue")
            .property("max-size-buffers", 3_u32)
            .property("max-size-bytes", 1_000_u32)
            .property("max-size-time", 10_000_000_000_u64)
            .build()
            .expect("constructing queue");
        let sink = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .property("signal-handoffs", true)
            .build()
            .expect("constructing blocking sink");
        let release_state = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let release = HandoffRelease(Arc::clone(&release_state));
        let first_handoff = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_handoff = Arc::clone(&first_handoff);
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        sink.connect("handoff", false, move |_values| {
            if !callback_handoff.swap(true, std::sync::atomic::Ordering::AcqRel) {
                if entered_tx.send(()).is_err() {
                    return None;
                }
                let (released, condition) = &*release_state;
                let mut released = released
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while !*released {
                    released = condition
                        .wait(released)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
            None
        });
        pipeline
            .add_many([&source, &queue, &sink])
            .expect("adding active queue elements");
        gst::Element::link_many([&source, &queue, &sink]).expect("linking active queue");
        pipeline
            .set_state(gst::State::Playing)
            .expect("starting queue pipeline");
        (pipeline, source, queue, sink, release, entered_rx)
    }

    #[test]
    fn tracer_properties_and_endpoint_lifecycle() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .property("max-pad-series", 7_u32)
            .build();
        assert!(tracer.property::<bool>("server-running"));
        assert_eq!(tracer.property::<u32>("max-pad-series"), 7);
        tracer.set_property("max-pad-series", 8_u32);
        assert_eq!(tracer.property::<u32>("max-pad-series"), 7);
        let address = tracer.property::<String>("bound-address");
        assert_ne!(address, "127.0.0.1:0");
        let response = scrape(&address);
        assert!(response.contains("200 OK"));
        assert!(response.ends_with("# EOF\n"));
        tracer.imp().dispose();
        tracer.imp().dispose();
        drop(tracer);
        let _error =
            std::net::TcpStream::connect(address).expect_err("endpoint must close after disposal");
    }

    #[test]
    fn tracer_invalid_configuration_remains_inactive() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        for (property, value) in [
            ("listen", "localhost:9099"),
            ("include-filter", "["),
            ("exclude-filter", "["),
        ] {
            let builder = glib::Object::builder::<super::super::PrometheusTracer>();
            let tracer = if property == "listen" {
                builder.property(property, value).build()
            } else {
                builder
                    .property("listen", "127.0.0.1:0")
                    .property(property, value)
                    .build()
            };
            assert!(!tracer.property::<bool>("server-running"));
            assert_eq!(tracer.property::<String>("bound-address"), "");
        }

        let occupied = TcpListener::bind("127.0.0.1:0").expect("binding occupied port");
        let address = occupied.local_addr().expect("occupied listener address");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", address.to_string())
            .build();
        assert!(!tracer.property::<bool>("server-running"));
    }

    #[test]
    fn tracer_hooks_feed_endpoint_without_graph_changes() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .build();
        assert!(tracer.property::<bool>("server-running"));
        let pipeline =
            gst::parse::launch("videotestsrc num-buffers=3 ! queue name=observed ! fakesink")
                .expect("constructing pipeline")
                .downcast::<gst::Pipeline>()
                .expect("launch returned a pipeline");
        assert_eq!(pipeline.children().len(), 3);
        pipeline
            .set_state(gst::State::Playing)
            .expect("starting pipeline");
        wait_for_eos(&pipeline);
        let response = scrape(&tracer.property::<String>("bound-address"));
        assert!(response.contains("gstsmith_gstreamer_pad_push_buffers_total"));
        assert!(response.contains("gstsmith_gstreamer_queue_level_buffers"));
        assert!(response.contains("gstsmith_gstreamer_pipeline_state"));
        pipeline
            .set_state(gst::State::Null)
            .expect("stopping pipeline");
    }

    #[test]
    fn tracer_buffer_list_hook_counts_members_and_bytes_exactly() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .property("include-filter", "list_element")
            .build();
        let pipeline = gst::parse::launch("identity name=list_element ! fakesink async=false")
            .expect("constructing buffer-list pipeline")
            .downcast::<gst::Pipeline>()
            .expect("launch returned a pipeline");
        pipeline
            .set_state(gst::State::Playing)
            .expect("starting buffer-list pipeline");
        let (result, _current, _pending) = pipeline.state(gst::ClockTime::from_seconds(5));
        result.expect("waiting for playing state");
        let element = pipeline.by_name("list_element").expect("finding identity");
        let source_pad = element.static_pad("src").expect("identity source pad");
        assert!(source_pad.push_event(gst::event::StreamStart::new("buffer-list-test")));
        let segment = gst::FormattedSegment::<gst::ClockTime>::new();
        assert!(source_pad.push_event(gst::event::Segment::new(&segment)));
        let mut list = gst::BufferList::new();
        let list_mut = list.get_mut().expect("unique buffer list");
        for size in [3, 5, 11] {
            list_mut.add(gst::Buffer::with_size(size).expect("allocating list member"));
        }
        source_pad.push_list(list).expect("pushing buffer list");

        let body = response_body(&scrape(&tracer.property::<String>("bound-address"))).to_owned();
        let labels = format!("{{element=\"{}\",pad=\"src\"}}", element.path_string());
        assert_sample(
            &body,
            &format!("gstsmith_gstreamer_pad_push_buffers_total{labels}"),
            "3",
        );
        assert_sample(
            &body,
            &format!("gstsmith_gstreamer_pad_push_bytes_total{labels}"),
            "19",
        );
        pipeline
            .set_state(gst::State::Null)
            .expect("stopping buffer-list pipeline");
    }

    #[test]
    fn tracer_filters_real_pad_queue_and_pipeline_hooks_consistently() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .property("include-filter", "allowed")
            .property("exclude-filter", "blocked")
            .build();
        let (allowed, allowed_source, allowed_queue) =
            finite_queue_pipeline("allowed_pipeline", "source", "queue");
        let (blocked, blocked_source, blocked_queue) =
            finite_queue_pipeline("allowed_blocked_pipeline", "source", "queue");
        for pipeline in [&allowed, &blocked] {
            pipeline
                .set_state(gst::State::Playing)
                .expect("starting filtered pipeline");
            wait_for_eos(pipeline);
        }

        let body = response_body(&scrape(&tracer.property::<String>("bound-address"))).to_owned();
        let allowed_pad = format!(
            "gstsmith_gstreamer_pad_push_buffers_total{{element=\"{}\",pad=\"src\"}}",
            allowed_source.path_string()
        );
        assert_sample(&body, &allowed_pad, "1");
        assert_sample(
            &body,
            "gstsmith_gstreamer_pipeline_state{pipeline=\"allowed_pipeline\",state=\"playing\"}",
            "1",
        );
        assert_sample(
            &body,
            &format!(
                "gstsmith_gstreamer_queue_level_buffers{{element=\"{}\"}}",
                allowed_queue.path_string()
            ),
            "0",
        );
        for blocked_identity in [
            blocked.name().to_string(),
            blocked_source.path_string().to_string(),
            blocked_queue.path_string().to_string(),
        ] {
            assert!(
                !body.contains(&blocked_identity),
                "{blocked_identity} in:\n{body}"
            );
        }
        for pipeline in [&allowed, &blocked] {
            pipeline
                .set_state(gst::State::Null)
                .expect("stopping filtered pipeline");
        }
    }

    #[test]
    fn tracer_real_hooks_limit_series_and_remove_dynamic_pads() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .property("include-filter", "limited_tee")
            .property("max-pad-series", 1_u32)
            .build();
        let pipeline = gst::Pipeline::builder().name("series_pipeline").build();
        let source = gst::ElementFactory::make("fakesrc")
            .property("num-buffers", 1_i32)
            .build()
            .expect("constructing series source");
        let tee = gst::ElementFactory::make("tee")
            .name("limited_tee")
            .build()
            .expect("constructing tee");
        let queue_a = gst::ElementFactory::make("queue")
            .build()
            .expect("constructing first branch queue");
        let queue_b = gst::ElementFactory::make("queue")
            .build()
            .expect("constructing second branch queue");
        let sink_a = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .build()
            .expect("constructing first branch sink");
        let sink_b = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .build()
            .expect("constructing second branch sink");
        pipeline
            .add_many([&source, &tee, &queue_a, &queue_b, &sink_a, &sink_b])
            .expect("adding series pipeline elements");
        source.link(&tee).expect("linking source to tee");
        queue_a.link(&sink_a).expect("linking first branch");
        queue_b.link(&sink_b).expect("linking second branch");
        let tee_pad_a = tee
            .request_pad_simple("src_%u")
            .expect("requesting first tee pad");
        let tee_pad_b = tee
            .request_pad_simple("src_%u")
            .expect("requesting second tee pad");
        tee_pad_a
            .link(&queue_a.static_pad("sink").expect("first queue sink pad"))
            .expect("linking first tee pad");
        tee_pad_b
            .link(&queue_b.static_pad("sink").expect("second queue sink pad"))
            .expect("linking second tee pad");
        pipeline
            .set_state(gst::State::Playing)
            .expect("starting series pipeline");
        wait_for_eos(&pipeline);

        let address = tracer.property::<String>("bound-address");
        let before = response_body(&scrape(&address)).to_owned();
        let tee_path = tee.path_string().to_string();
        let tracked = before
            .lines()
            .filter(|line| {
                line.starts_with("gstsmith_gstreamer_pad_push_buffers_total{")
                    && line.contains(&format!("element=\"{tee_path}\""))
            })
            .collect::<Vec<_>>();
        assert_eq!(tracked.len(), 1, "{before}");
        assert!(tracked[0].ends_with(" 1"), "{before}");
        assert_sample(
            &before,
            "gstsmith_gstreamer_untracked_pad_events_total{reason=\"series_limit\"}",
            "1",
        );

        pipeline
            .set_state(gst::State::Null)
            .expect("stopping series pipeline");
        tee.release_request_pad(&tee_pad_a);
        tee.release_request_pad(&tee_pad_b);
        let after = response_body(&scrape(&address)).to_owned();
        assert!(
            !after.lines().any(|line| {
                line.starts_with("gstsmith_gstreamer_pad_push_buffers_total{")
                    && line.contains(&format!("element=\"{tee_path}\""))
            }),
            "{after}"
        );
    }

    #[test]
    fn tracer_real_queue_hooks_refresh_values_and_remove_entries() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .build();
        let (pipeline, source, queue, sink, release, entered_rx) = blocked_queue_pipeline();
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first buffer reached blocking sink");
        let level_wait_started = std::time::Instant::now();
        while queue.property::<u32>("current-level-buffers") != 3 {
            assert!(level_wait_started.elapsed() < Duration::from_secs(2));
            std::thread::yield_now();
        }
        let queue2 = gst::ElementFactory::make("queue2")
            .name("observed_queue2")
            .property("max-size-buffers", 17_u32)
            .property("max-size-bytes", 31_u32)
            .property("max-size-time", 1_500_000_000_u64)
            .build()
            .expect("constructing queue2");
        let multiqueue = gst::ElementFactory::make("multiqueue")
            .name("ignored_multiqueue")
            .build()
            .expect("constructing multiqueue");
        pipeline
            .add_many([&queue2, &multiqueue])
            .expect("adding comparison queue elements");
        let queue_path = queue.path_string().to_string();
        let queue2_path = queue2.path_string().to_string();
        let multiqueue_path = multiqueue.path_string().to_string();
        let address = tracer.property::<String>("bound-address");
        let body = response_body(&scrape(&address)).to_owned();
        assert_eq!(queue.property::<u32>("current-level-buffers"), 3);
        assert_eq!(queue.property::<u32>("current-level-bytes"), 300);
        assert_eq!(queue.property::<u64>("current-level-time"), 3_000_000_000);
        release.release();
        wait_for_eos(&pipeline);
        pipeline
            .set_state(gst::State::Null)
            .expect("stopping queue pipeline");
        for (path, buffers, bytes, seconds) in [
            (&queue_path, "3", "1000", "10.0"),
            (&queue2_path, "17", "31", "1.5"),
        ] {
            let (level_buffers, level_bytes, level_seconds) = if path == &queue_path {
                ("3", "300", "3.0")
            } else {
                ("0", "0", "0.0")
            };
            for (metric, value) in [
                ("queue_level_buffers", level_buffers),
                ("queue_level_bytes", level_bytes),
                ("queue_level_seconds", level_seconds),
                ("queue_capacity_buffers", buffers),
                ("queue_capacity_bytes", bytes),
                ("queue_capacity_seconds", seconds),
            ] {
                assert_sample(
                    &body,
                    &format!("gstsmith_gstreamer_{metric}{{element=\"{path}\"}}"),
                    value,
                );
            }
        }
        assert!(!body.contains(&multiqueue_path), "{body}");

        source.unlink(&queue);
        queue.unlink(&sink);
        pipeline.remove(&queue).expect("removing queue");
        pipeline.remove(&queue2).expect("removing queue2");
        let after = response_body(&scrape(&address)).to_owned();
        for path in [&queue_path, &queue2_path] {
            assert!(
                !after.lines().any(|line| {
                    line.starts_with("gstsmith_gstreamer_queue_")
                        && line.contains(&format!("element=\"{path}\""))
                }),
                "{after}"
            );
        }
    }

    #[test]
    fn tracer_pipeline_state_matches_actual_successful_transitions() {
        let _guard = TRACER_TEST_LOCK.lock();
        gst::init().expect("initializing GStreamer");
        let tracer = glib::Object::builder::<super::super::PrometheusTracer>()
            .property("listen", "127.0.0.1:0")
            .build();
        let pipeline = gst::Pipeline::builder()
            .name("actual_state_pipeline")
            .build();
        let source = gst::ElementFactory::make("videotestsrc")
            .property("is-live", true)
            .build()
            .expect("constructing live source");
        let sink = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .build()
            .expect("constructing state sink");
        pipeline
            .add_many([&source, &sink])
            .expect("adding state pipeline elements");
        source.link(&sink).expect("linking state pipeline");
        let address = tracer.property::<String>("bound-address");
        for state in [gst::State::Playing, gst::State::Paused, gst::State::Ready] {
            pipeline.set_state(state).expect("changing pipeline state");
            let (result, current, pending) = pipeline.state(gst::ClockTime::from_seconds(5));
            result.expect("waiting for pipeline state");
            assert_eq!(current, state);
            assert_eq!(pending, gst::State::VoidPending);
            let body = response_body(&scrape(&address)).to_owned();
            let active = match state {
                gst::State::Null => "null",
                gst::State::Ready => "ready",
                gst::State::Paused => "paused",
                gst::State::Playing => "playing",
                gst::State::VoidPending => "void-pending",
            };
            for candidate in ["null", "ready", "paused", "playing"] {
                assert_sample(
                    &body,
                    &format!(
                        "gstsmith_gstreamer_pipeline_state{{pipeline=\"actual_state_pipeline\",state=\"{candidate}\"}}"
                    ),
                    if candidate == active { "1" } else { "0" },
                );
            }
        }
        pipeline
            .set_state(gst::State::Null)
            .expect("stopping state pipeline");
    }
}
