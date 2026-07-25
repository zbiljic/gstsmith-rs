use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};

static RUNTIME: LazyLock<Result<Runtime, std::io::Error>> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("s2-gst-worker")
        .enable_all()
        .build()
});

pub fn runtime() -> Result<&'static Runtime, String> {
    RUNTIME
        .as_ref()
        .map_err(|error| format!("failed to create the S2 runtime: {error}"))
}
