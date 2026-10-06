# Pipeline delivery spacing, latency, and queue history

This finite example combines this workspace's Prometheus tracer with
GStreamer's existing latency and queue-level tracers. Run from the
`gstsmith-rs` workspace root. Build the plugin and check the native tracers:

```sh
cargo build -p gst-plugin-prometheus
gst-inspect-1.0 latency
gst-inspect-1.0 queue-levels
```

`latency` comes from GStreamer's `coretracers`; `queue-levels` comes from
`gst-plugins-rs`'s `rstracers` plugin. Install those plugins if inspection fails.
The workspace does not build or bundle them.

The source aims for 30 frames/sec, while `work` sleeps 40 ms per buffer. The
queue therefore fills and eventually applies backpressure. This makes the
different measurements visible without requiring a video window:

```sh
trace_dir=$(mktemp -d "${TMPDIR:-/tmp}/gstsmith-tracing.XXXXXX")
printf 'Trace output: %s\n' "$trace_dir"

GST_PLUGIN_PATH="$PWD/target/debug${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}" \
GST_TRACERS="latency(flags=pipeline+element+reported);queue-levels(file=\"$trace_dir/queues.csv\");prometheus(listen=\"127.0.0.1:9099\",track-intervals=(boolean)true)" \
GST_DEBUG=GST_TRACER:7 \
GST_DEBUG_NO_COLOR=1 \
GST_DEBUG_FILE="$trace_dir/latency.log" \
gst-launch-1.0 \
    videotestsrc \
        is-live=true num-buffers=300 \
    ! video/x-raw,width=320,height=240,framerate=30/1 \
    ! queue \
        name=backlog max-size-buffers=8 max-size-bytes=0 max-size-time=0 \
    ! identity \
        name=work sleep-time=40000 \
    ! fakesink \
        sync=false
```

While it runs (roughly 12 seconds), scrape from another terminal:

```sh
curl --fail http://127.0.0.1:9099/metrics
```

| Output | What to inspect |
|---|---|
| `latency.log` | Native pipeline/per-element transit timings and reported latency. Event correlation can depend on element behavior; these are not CPU measurements. |
| `queues.csv` | Queue occupancy over time, including configured limits. Written by the native queue-levels tracer. |
| Prometheus endpoint | `gstsmith_gstreamer_pad_push_interval_seconds` histogram, buffer/byte counters, and current queue gauges. The endpoint stops when the pipeline process exits. |

For a StatsD receiver, build `gst-plugin-statsd` and replace the `prometheus(...)`
entry with `statsd(destination="127.0.0.1:8125",track-intervals=(boolean)true)`.
See its [wire format](../../utils/statsd/README.md#push-arrival-intervals): interval
buckets are counter deltas, with millisecond bounds and a nanosecond sum.

Reference: native [latency](https://gstreamer.freedesktop.org/documentation/coretracers/latency.html)
and [queue-levels](https://gstreamer.freedesktop.org/documentation/rstracers/queue-levels.html)
documentation. These tracers complement the
[Prometheus metrics](../../utils/prometheus/README.md#push-arrival-intervals).
