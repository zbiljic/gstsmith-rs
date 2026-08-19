use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};

static RUNTIME: LazyLock<Result<Runtime, std::io::Error>> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name_fn({
            let next = std::sync::atomic::AtomicUsize::new(1);
            move || {
                let id = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                format!("gst-vlm-{id}")
            }
        })
        .enable_all()
        .build()
});

pub(crate) fn runtime() -> Result<&'static Runtime, String> {
    RUNTIME
        .as_ref()
        .map_err(|error| format!("failed to create the VLM runtime: {error}"))
}
