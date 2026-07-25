use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};

static RUNTIME: LazyLock<Result<Runtime, std::io::Error>> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("gst-nats")
        .enable_all()
        .build()
});

pub fn runtime() -> Result<&'static Runtime, String> {
    RUNTIME
        .as_ref()
        .map_err(|error| format!("failed to create the NATS runtime: {error}"))
}
