use std::convert::Infallible;
use std::future::{Future, poll_fn};
use std::net::{SocketAddr, TcpListener};
use std::pin::pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::task::Poll;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::header::{ALLOW, CACHE_CONTROL, CONTENT_TYPE};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::sync::watch;

use crate::metrics::Metrics;

const CONTENT_TYPE_VALUE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";
const CONNECTION_LIFETIME: Duration = Duration::from_secs(30);
const CONNECTION_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

static CAT: std::sync::LazyLock<gst::DebugCategory> = std::sync::LazyLock::new(|| {
    gst::DebugCategory::new(
        "prometheus-server",
        gst::DebugColorFlags::empty(),
        Some("Prometheus metrics HTTP server"),
    )
});

enum ShutdownOr<T> {
    Shutdown,
    Output(T),
}

async fn select_shutdown<F>(
    shutdown: &mut watch::Receiver<bool>,
    future: F,
) -> ShutdownOr<F::Output>
where
    F: Future,
{
    if *shutdown.borrow() {
        return ShutdownOr::Shutdown;
    }
    let mut changed = pin!(shutdown.changed());
    let mut future = pin!(future);
    poll_fn(|context| {
        if changed.as_mut().poll(context).is_ready() {
            Poll::Ready(ShutdownOr::Shutdown)
        } else {
            future.as_mut().poll(context).map(ShutdownOr::Output)
        }
    })
    .await
}

pub(crate) struct ServerHandle {
    pub(crate) address: SocketAddr,
    shutdown: watch::Sender<bool>,
    thread: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    active_connections: Arc<AtomicUsize>,
}

impl ServerHandle {
    pub(crate) fn stop(&mut self) -> Result<(), String> {
        let _receiver_count = self.shutdown.send_replace(true);
        let join_result = self.thread.take().map_or(Ok(()), |thread| {
            thread.join().map_err(|_panic_payload| {
                "Prometheus server thread terminated unexpectedly".to_owned()
            })
        });
        self.running.store(false, Ordering::Release);
        join_result?;
        if self.active_connections.load(Ordering::Acquire) != 0 {
            return Err("Prometheus connections did not stop cleanly".to_owned());
        }
        Ok(())
    }

    pub(crate) fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Acquire)
    }
}

struct ActiveConnectionGuard(Arc<AtomicUsize>);

impl Drop for ActiveConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) fn start(listener: TcpListener, metrics: Arc<Metrics>) -> Result<ServerHandle, String> {
    let address = listener
        .local_addr()
        .map_err(|error| format!("failed to read bound address: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("failed to configure listener: {error}"))?;
    let (shutdown, shutdown_rx) = watch::channel(false);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let running = Arc::new(AtomicBool::new(false));
    let thread_running = Arc::clone(&running);
    let active_connections = Arc::new(AtomicUsize::new(0));
    let thread_active_connections = Arc::clone(&active_connections);
    let thread = thread::Builder::new()
        .name("gst-prometheus-http".to_owned())
        .spawn(move || {
            run(
                listener,
                metrics,
                shutdown_rx,
                ready_tx,
                thread_running,
                thread_active_connections,
            );
        })
        .map_err(|error| format!("failed to start server thread: {error}"))?;

    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => Ok(ServerHandle {
            address,
            shutdown,
            thread: Some(thread),
            running,
            active_connections,
        }),
        Ok(Err(error)) => {
            let _receiver_count = shutdown.send_replace(true);
            match thread.join() {
                Ok(()) => Err(error),
                Err(_panic_payload) => Err(format!("{error}; server thread did not join cleanly")),
            }
        }
        Err(error) => {
            let _receiver_count = shutdown.send_replace(true);
            let message = format!("server startup handshake failed: {error}");
            match thread.join() {
                Ok(()) => Err(message),
                Err(_panic_payload) => {
                    Err(format!("{message}; server thread did not join cleanly"))
                }
            }
        }
    }
}

fn run(
    listener: TcpListener,
    metrics: Arc<Metrics>,
    shutdown: watch::Receiver<bool>,
    ready: mpsc::SyncSender<Result<(), String>>,
    running: Arc<AtomicBool>,
    active_connections: Arc<AtomicUsize>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            if ready
                .send(Err(format!("failed to create server runtime: {error}")))
                .is_err()
            {
                gst::warning!(
                    CAT,
                    "Prometheus startup receiver closed before runtime failure"
                );
            }
            return;
        }
    };
    runtime.block_on(async move {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(error) => {
                if ready
                    .send(Err(format!("failed to adopt listener: {error}")))
                    .is_err()
                {
                    gst::warning!(
                        CAT,
                        "Prometheus startup receiver closed before listener failure"
                    );
                }
                return;
            }
        };
        running.store(true, Ordering::Release);
        if ready.send(Ok(())).is_err() {
            running.store(false, Ordering::Release);
            return;
        }

        serve_connections(listener, metrics, shutdown, active_connections).await;
        running.store(false, Ordering::Release);
    });
}

async fn serve_connections(
    listener: tokio::net::TcpListener,
    metrics: Arc<Metrics>,
    mut shutdown: watch::Receiver<bool>,
    active_connections: Arc<AtomicUsize>,
) {
    let mut connections = tokio::task::JoinSet::new();
    loop {
        match select_shutdown(&mut shutdown, listener.accept()).await {
            ShutdownOr::Shutdown => break,
            ShutdownOr::Output(Ok((stream, _peer))) => spawn_connection(
                &mut connections,
                stream,
                Arc::clone(&metrics),
                shutdown.clone(),
                Arc::clone(&active_connections),
            ),
            ShutdownOr::Output(Err(error)) => {
                gst::error!(CAT, "Prometheus listener accept failed: {error}");
                break;
            }
        }
        while let Some(completed) = connections.try_join_next() {
            if completed.is_err() {
                gst::warning!(
                    CAT,
                    "Prometheus HTTP connection task terminated unexpectedly"
                );
            }
        }
    }
    drain_connections(&mut connections).await;
}

fn spawn_connection(
    connections: &mut tokio::task::JoinSet<()>,
    stream: tokio::net::TcpStream,
    metrics: Arc<Metrics>,
    mut shutdown: watch::Receiver<bool>,
    active_connections: Arc<AtomicUsize>,
) {
    active_connections.fetch_add(1, Ordering::AcqRel);
    let active_guard = ActiveConnectionGuard(active_connections);
    connections.spawn(async move {
        let _active_guard = active_guard;
        let io = TokioIo::new(stream);
        let service = service_fn(move |request| {
            std::future::ready(Ok::<_, Infallible>(respond(&request, &metrics)))
        });
        let connection = http1::Builder::new()
            .keep_alive(false)
            .serve_connection(io, service);
        match select_shutdown(
            &mut shutdown,
            tokio::time::timeout(CONNECTION_LIFETIME, connection),
        )
        .await
        {
            ShutdownOr::Shutdown | ShutdownOr::Output(Ok(Ok(()))) => {}
            ShutdownOr::Output(Ok(Err(_error))) => gst::warning!(
                CAT,
                "Prometheus HTTP connection ended with a protocol error"
            ),
            ShutdownOr::Output(Err(_elapsed)) => {
                gst::warning!(CAT, "Prometheus HTTP connection exceeded its lifetime");
            }
        }
    });
}

async fn drain_connections(connections: &mut tokio::task::JoinSet<()>) {
    while !connections.is_empty() {
        match tokio::time::timeout(CONNECTION_SHUTDOWN_TIMEOUT, connections.join_next()).await {
            Ok(Some(Ok(()))) => {}
            Ok(Some(Err(_error))) => gst::warning!(
                CAT,
                "Prometheus HTTP connection task terminated unexpectedly during shutdown"
            ),
            Ok(None) => break,
            Err(_elapsed) => {
                gst::warning!(
                    CAT,
                    "Prometheus HTTP connections exceeded shutdown deadline"
                );
                connections.abort_all();
                break;
            }
        }
    }
}

fn respond(request: &Request<Incoming>, metrics: &Metrics) -> Response<Full<Bytes>> {
    let path = request.uri().path();
    let method = request.method();
    if path != "/metrics" {
        build_response(StatusCode::NOT_FOUND, Bytes::new(), false, false)
    } else if method == Method::GET || method == Method::HEAD {
        match metrics.encode() {
            Ok(output) => {
                let body = if method == Method::HEAD {
                    Bytes::new()
                } else {
                    Bytes::from(output)
                };
                build_response(StatusCode::OK, body, true, false)
            }
            Err(_error) => {
                gst::error!(CAT, "Prometheus metric encoding failed");
                build_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Bytes::from_static(b"metric encoding failed\n"),
                    false,
                    false,
                )
            }
        }
    } else {
        build_response(StatusCode::METHOD_NOT_ALLOWED, Bytes::new(), false, true)
    }
}

fn build_response(
    status: StatusCode,
    body: Bytes,
    openmetrics: bool,
    allow: bool,
) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(body));
    *response.status_mut() = status;
    if openmetrics {
        response.headers_mut().insert(
            CONTENT_TYPE,
            hyper::header::HeaderValue::from_static(CONTENT_TYPE_VALUE),
        );
        response.headers_mut().insert(
            CACHE_CONTROL,
            hyper::header::HeaderValue::from_static("no-store"),
        );
    }
    if allow {
        response
            .headers_mut()
            .insert(ALLOW, hyper::header::HeaderValue::from_static("GET, HEAD"));
    }
    response
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::sync::Barrier;

    use gst::prelude::*;

    use super::*;

    fn request(address: SocketAddr, request: &[u8]) -> String {
        let mut stream = std::net::TcpStream::connect(address).expect("connecting");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("setting read timeout");
        stream.write_all(request).expect("writing request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("reading response");
        response
    }

    fn response_parts(response: &str) -> (&str, &str) {
        response
            .split_once("\r\n\r\n")
            .expect("HTTP response has a header terminator")
    }

    fn header<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
        headers.lines().skip(1).find_map(|line| {
            let (header_name, value) = line.split_once(": ")?;
            header_name.eq_ignore_ascii_case(name).then_some(value)
        })
    }

    #[test]
    fn server_routes_get_head_not_found_and_method_not_allowed() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding test listener");
        let metrics = Metrics::new(None, None, 1);
        let mut server = start(listener, metrics).expect("starting test server");

        let get = request(
            server.address,
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        let (get_headers, get_body) = response_parts(&get);
        assert_eq!(get_headers.lines().next(), Some("HTTP/1.1 200 OK"));
        assert_eq!(
            header(get_headers, "content-type"),
            Some(CONTENT_TYPE_VALUE)
        );
        assert_eq!(header(get_headers, "cache-control"), Some("no-store"));
        assert!(get_body.ends_with("# EOF\n"));

        let head = request(
            server.address,
            b"HEAD /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        let (head_headers, head_body) = response_parts(&head);
        assert_eq!(head_headers.lines().next(), Some("HTTP/1.1 200 OK"));
        assert_eq!(
            header(head_headers, "content-type"),
            Some(CONTENT_TYPE_VALUE)
        );
        assert_eq!(header(head_headers, "cache-control"), Some("no-store"));
        assert_eq!(head_body, "");

        let missing = request(
            server.address,
            b"GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(
            response_parts(&missing).0.lines().next(),
            Some("HTTP/1.1 404 Not Found")
        );

        let method = request(
            server.address,
            b"POST /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        let (method_headers, _body) = response_parts(&method);
        assert_eq!(
            method_headers.lines().next(),
            Some("HTTP/1.1 405 Method Not Allowed")
        );
        assert_eq!(header(method_headers, "allow"), Some("GET, HEAD"));
        server.stop().expect("stopping server");
    }

    #[test]
    fn shutdown_interrupts_partial_client() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding test listener");
        let metrics = Metrics::new(None, None, 1);
        let mut server = start(listener, metrics).expect("starting test server");
        let mut stream = std::net::TcpStream::connect(server.address).expect("connecting");
        stream
            .write_all(b"GET /metrics")
            .expect("writing partial request");
        let wait_started = std::time::Instant::now();
        while server.active_connections() == 0 {
            assert!(wait_started.elapsed() < Duration::from_secs(2));
            std::thread::yield_now();
        }
        let started = std::time::Instant::now();
        server.stop().expect("stopping server");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(server.active_connections(), 0);
    }

    #[test]
    fn encoding_failure_returns_500_and_is_counted_on_next_scrape() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding test listener");
        let metrics = Metrics::new(None, None, 1);
        metrics.fail_next_encoding_for_test();
        let mut server = start(listener, Arc::clone(&metrics)).expect("starting test server");

        let failure = request(
            server.address,
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        let (failure_headers, failure_body) = response_parts(&failure);
        assert_eq!(
            failure_headers.lines().next(),
            Some("HTTP/1.1 500 Internal Server Error")
        );
        assert_eq!(failure_body, "metric encoding failed\n");

        let recovered = request(
            server.address,
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert!(
            response_parts(&recovered)
                .1
                .lines()
                .any(|line| line == "gstsmith_gstreamer_scrape_encoding_failures_total 1")
        );
        server.stop().expect("stopping server");
    }

    #[test]
    fn simultaneous_updates_and_http_scrapes_preserve_exact_totals() {
        gst::init().expect("initializing GStreamer");
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding test listener");
        let metrics = Metrics::new(None, None, 1);
        let element = gst::ElementFactory::make("identity")
            .name("concurrent-source")
            .build()
            .expect("constructing concurrent element");
        let pad = element.static_pad("src").expect("concurrent source pad");
        metrics.update_pad(&pad, 0, 0);
        let mut server = start(listener, Arc::clone(&metrics)).expect("starting test server");
        let barrier = Arc::new(Barrier::new(6));
        let workers = (0..4)
            .map(|_| {
                let metrics = Arc::clone(&metrics);
                let pad = pad.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..1_000 {
                        metrics.update_pad(&pad, 1, 10);
                    }
                })
            })
            .collect::<Vec<_>>();
        let address = server.address;
        let scrape_barrier = Arc::clone(&barrier);
        let scraper = std::thread::spawn(move || {
            scrape_barrier.wait();
            for _ in 0..50 {
                let response =
                    request(address, b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
                assert!(response_parts(&response).1.ends_with("# EOF\n"));
            }
        });
        barrier.wait();
        for worker in workers {
            worker.join().expect("counter worker");
        }
        scraper.join().expect("scrape worker");

        let response = request(
            server.address,
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        let body = response_parts(&response).1;
        let element_path = element.path_string();
        assert!(body.lines().any(|line| {
            line == format!(
                "gstsmith_gstreamer_pad_push_buffers_total{{element=\"{element_path}\",pad=\"src\"}} 4000"
            )
        }), "{body}");
        assert!(body.lines().any(|line| {
            line == format!(
                "gstsmith_gstreamer_pad_push_bytes_total{{element=\"{element_path}\",pad=\"src\"}} 40000"
            )
        }), "{body}");
        server.stop().expect("stopping server");
    }

    #[test]
    fn client_disconnect_does_not_break_subsequent_scrapes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding test listener");
        let metrics = Metrics::new(None, None, 1);
        let mut server = start(listener, metrics).expect("starting test server");
        let mut disconnected =
            std::net::TcpStream::connect(server.address).expect("connecting disconnecting client");
        disconnected
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("writing disconnected request");
        drop(disconnected);

        let response = request(
            server.address,
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(
            response_parts(&response).0.lines().next(),
            Some("HTTP/1.1 200 OK")
        );
        server.stop().expect("stopping server");
    }

    #[test]
    fn stop_clears_running_after_server_thread_failure() {
        let (shutdown, _shutdown_rx) = watch::channel(false);
        let running = Arc::new(AtomicBool::new(true));
        let failed_thread = std::thread::spawn(|| {
            std::panic::resume_unwind(Box::new("injected server thread failure"));
        });
        let mut server = ServerHandle {
            address: "127.0.0.1:0".parse().expect("test socket address"),
            shutdown,
            thread: Some(failed_thread),
            running,
            active_connections: Arc::new(AtomicUsize::new(0)),
        };

        assert!(server.stop().is_err());
        assert!(!server.is_running());
    }
}
