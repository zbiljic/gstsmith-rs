# GStreamer Prometheus tracer

The `prometheus` tracer observes all GStreamer pipelines in its process without
changing their graphs. It keeps buffer hooks to bounded counter updates and
serves an OpenMetrics 1.0 snapshot from a dedicated HTTP thread. It requires
GStreamer 1.24 or newer.

Enable the tracer before starting a pipeline:

```sh
GST_TRACERS='prometheus(listen="127.0.0.1:9099")' \
gst-launch-1.0 \
  videotestsrc is-live=true \
  ! queue \
  ! fakesink
```

Scrape the endpoint from another shell:

```sh
curl --fail http://127.0.0.1:9099/metrics
```

## Properties

| Property | Default | Meaning |
|---|---:|---|
| `listen` | `127.0.0.1:9099` | Numeric listener address; port zero selects an available port |
| `include-filter` | unset | Regular expression selecting pad, queue, and pipeline identities |
| `exclude-filter` | unset | Regular expression applied after the include filter |
| `max-pad-series` | `256` | Maximum active pad label sets (1 through 65535) |
| `bound-address` | - | Read-only actual listener address; empty when startup fails |
| `server-running` | - | Read-only server startup status |

Configuration properties are startup-only; later writes are ignored. Use
GStreamer structure typing for the unsigned series limit, for example
`max-pad-series=(uint)512` inside `GST_TRACERS`.

Invalid addresses, invalid regular expressions, and bind failures leave the
tracer alive but inactive. Because GStreamer creates tracers during process
startup, applications should inspect the tracer properties or logs when they
need to detect a startup error.

## Metrics

| Metric | Type | Labels |
|---|---|---|
| `gstsmith_gstreamer_pad_push_buffers_total` | counter | `element`, `pad` |
| `gstsmith_gstreamer_pad_push_bytes_total` | counter | `element`, `pad` |
| `gstsmith_gstreamer_pipeline_state` | gauge | `pipeline`, `state` |
| `gstsmith_gstreamer_queue_level_buffers` | gauge | `element` |
| `gstsmith_gstreamer_queue_level_bytes` | gauge | `element` |
| `gstsmith_gstreamer_queue_level_seconds` | gauge | `element` |
| `gstsmith_gstreamer_queue_capacity_buffers` | gauge | `element` |
| `gstsmith_gstreamer_queue_capacity_bytes` | gauge | `element` |
| `gstsmith_gstreamer_queue_capacity_seconds` | gauge | `element` |
| `gstsmith_gstreamer_untracked_pad_events_total` | counter | `reason` |
| `gstsmith_gstreamer_scrape_encoding_failures_total` | counter | none |

The push counters represent attempts, not successful downstream delivery. A
buffer list contributes its member count and the sum of member sizes. Queue
gauges cover the core `queue` and `queue2` elements and are refreshed at scrape
time. Prometheus derives rates at the desired window:

```promql
rate(gstsmith_gstreamer_pad_push_buffers_total[1m])
```

```promql
rate(gstsmith_gstreamer_pad_push_bytes_total[1m]) * 8
```

## Scope and security

The endpoint serves plaintext HTTP without authentication. Keep the default
loopback listener, or place a non-loopback listener on a trusted network or
behind an authenticating TLS reverse proxy.

The tracer reports only the pipeline, `queue`/`queue2`, and pad-push metrics
listed above. It does not collect host metrics, end-to-end latency, or metrics
for `multiqueue` and `appsrc`. Each process needs its own port; deployment labels
belong in Prometheus scrape configuration.
