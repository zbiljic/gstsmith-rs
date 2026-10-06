# StatsD metrics tracer

The `statsd` tracer observes GStreamer pipelines without changing their graphs
and periodically pushes bounded metrics over UDP. Despite the familiar StatsD
name, this first version requires a server that accepts DogStatsD-style `|#`
tags. A Datadog Agent or a correctly configured Prometheus `statsd_exporter`
are suitable receivers; compatibility with every classic StatsD daemon is not
claimed.

## Example

The default destination is the loopback address `127.0.0.1:8125`:

```sh
GST_TRACERS='statsd(global-tags="service:video,env:dev")' \
gst-launch-1.0 \
    videotestsrc \
        num-buffers=30 \
    ! queue \
        name=observed \
    ! fakesink \
        sync=false
```

Configuration is applied when the tracer is created. Later property writes are
ignored with a warning.

## Properties

| Property | Type | Default | Description |
|---|---|---:|---|
| `destination` | string | `127.0.0.1:8125` | Numeric IPv4 or IPv6 UDP destination; DNS names are rejected. |
| `prefix` | string | `gstsmith` | Non-empty ASCII metric prefix, at most 128 bytes. |
| `global-tags` | nullable string | unset | Up to 16 comma-separated `key:value` tags and 512 bytes. |
| `flush-interval-ms` | unsigned integer | `1000` | Worker interval from 100 through 60000 milliseconds. |
| `include-filter` | nullable string | unset | Regex selecting unsanitized metric scope identities. |
| `exclude-filter` | nullable string | unset | Regex applied after include; matching scopes are excluded. |
| `max-pad-series` | unsigned integer | `256` | Exact active pad-labelset cap from 1 through 65535. |
| `track-intervals` | boolean | `false` | Collect fixed-bucket histograms of spacing between pad push attempts. |
| `worker-running` | read-only boolean | `false` | True only after worker startup succeeds. |

Invalid destinations, prefixes, tags, or regular expressions leave the tracer
inactive. Check `worker-running` and GStreamer logs when no metrics arrive.

## Metrics

Cadence joins `prefix` and each key below with `.`. With the default prefix,
metric names begin with `gstsmith.gstreamer`. Counter values are interval
deltas rather than cumulative process totals.

| Key after prefix | Type | Dynamic tags | Meaning |
|---|---|---|---|
| `gstreamer.pad.push_buffers` | counter | `element`, `pad` | Buffer attempts since the last accepted emission. |
| `gstreamer.pad.push_bytes` | counter | `element`, `pad` | Byte attempts since the last accepted emission. |
| `gstreamer.pad.push_interval.bucket` | counter, opt-in | `element`, `pad`, `le_ms` | New intervals at or below the given upper bound in milliseconds. |
| `gstreamer.pad.push_interval.count` | counter, opt-in | `element`, `pad` | Number of new interval samples. |
| `gstreamer.pad.push_interval.sum_ns` | counter, opt-in | `element`, `pad` | Sum of new intervals in nanoseconds. |
| `gstreamer.pipeline.state` | gauge | `pipeline`, `state` | One-hot `null`, `ready`, `paused`, and `playing` values. |
| `gstreamer.queue.level_buffers` | gauge | `element` | Current queued buffers. |
| `gstreamer.queue.level_bytes` | gauge | `element` | Current queued bytes. |
| `gstreamer.queue.level_seconds` | gauge | `element` | Current queued time in seconds. |
| `gstreamer.queue.capacity_buffers` | gauge | `element` | Configured maximum buffers. |
| `gstreamer.queue.capacity_bytes` | gauge | `element` | Configured maximum bytes. |
| `gstreamer.queue.capacity_seconds` | gauge | `element` | Configured maximum time in seconds. |
| `gstreamer.untracked_pad_events` | counter | `reason=series_limit` | Attempts excluded by the active pad cap. |
| `gstreamer.statsd_export_errors` | counter | `reason=emit\|flush` | Worker emission and flush failures. |
| `gstreamer.statsd_dropped_series` | counter | `reason=retirement_queue_full` | Retired series without a final-drain handoff. |

Pad counters describe `pad-push-pre` attempts, not successful downstream
delivery. After a positive counter delta, the worker emits one idle zero and
then suppresses repeated idle zeroes. Retired pads receive a final best-effort
drain. UDP socket acceptance is not proof of delivery or ingestion, and the
transport does not provide exactly-once semantics.

GStreamer-derived tag values are limited to 192 ASCII bytes. ASCII letters,
digits, `_`, `-`, `.`, and `/` are preserved; other Unicode scalars, controls,
and DogStatsD delimiters become `_`. Filters always see the original identity
before sanitization. Dynamic object identities are never placed in metric
names.

Queue identities and filters refresh after bin additions/removals and renames.
Direct `GstObject` parenting outside bin APIs is detected by a fallback check at
the first export at least 30 seconds after the previous check. Queue levels
and configured limits remain current at each export.

## Push arrival intervals

Set `track-intervals=(boolean)true` in the tracer parameters. Intervals use
GStreamer's monotonic hook timestamps. Each buffer or buffer-list push is one
arrival; list members do not create extra samples. The first arrival establishes
a baseline. Flush, stream-start, segment, EOS, and element state transitions
clear that baseline while preserving accumulated counts. Concurrent hooks
observed out of timestamp order are skipped for interval measurement.

The fixed upper bounds are 1, 5, 10, 20, 40, 100, 250, and 1000 ms, plus `+Inf`.
Buckets are cumulative across bounds: a 4 ms interval increments every bucket
from `le_ms:5` through `le_ms:+Inf`. The worker exports temporal deltas of these
counts, using ordinary DogStatsD **counters**. They are pre-aggregated buckets,
not DogStatsD `h`/`d` samples; receiver-generated histogram percentiles are not
automatically available. The mean in milliseconds is `sum_ns / count / 1e6`
using totals over the same time window. Unobserved buckets emit nothing;
interpret their missing counters as zero.

Histograms share existing pad filters, series limits, and retirement handling.
Their state is bounded per tracked pad and stores no individual samples.
They measure push-attempt spacing, not processing time or transit latency.
See the [combined tracing example](../../examples/tracing/README.md) for
complementary latency and queue diagnostics.

## Runtime and scope

Recurring counter updates use a cached Papaya lookup and relaxed atomics.
Opt-in interval collection also takes a short per-pad lock. The
owned `gst-statsd-export` thread performs formatting, queue inspection,
buffering, and all network work. It uses Cadence's buffered UDP sink, not
Tokio, a global metrics recorder, DNS, or an additional queue. Disposal wakes
and joins the worker and attempts one final flush.

This version deliberately excludes TCP, Unix sockets, sampling, persistent
retries, disk spooling, TLS, authentication, dashboards, host metrics,
`multiqueue`, queue overruns, pull-mode counters, and push-post failure
correlation. Latency, processing-time, and queue-overrun measurements are
possible separately measured follow-ups rather than partially supported
features here.
