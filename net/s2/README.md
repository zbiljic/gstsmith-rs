# S2 GStreamer plugin

This crate provides native, single-stream transports for
[S2](https://s2.dev/):

- `s2src` continuously reads one configured stream and emits one arbitrary
  byte buffer per S2 record.
- `s2sink` appends one arbitrary byte buffer per S2 record and drains accepted
  records at EOS or a normal state stop.

Both pads advertise `ANY`. Empty and non-UTF-8 record bodies are preserved.
The elements do not create basins or streams.

## Integration tests

The normal `mise run pre-commit` gate is offline/local and does not require
Docker. To run this crate's ignored S2 Lite integration suite, ensure a
working Docker daemon is available and run:

```sh
mise run test:integration:s2
```

The task pulls the pinned S2 Lite image with bounded retries before starting
the serial test binary. `mise run test:integration` requires both Docker and a
reachable Core NATS server.

## Authentication and endpoints

Set `basin` and `stream` on each element. By default, credentials come from
`S2_ACCESS_TOKEN`. For managed deployments, prefer a mode-appropriate secret
file and set `access-token-file`; its value takes precedence over the
environment. Tokens and fencing values are intentionally not GObject
properties because properties are introspectable.

These examples assume `mise run build` has completed from the repository root.

Cloud sink example:

```sh
S2_ACCESS_TOKEN='...' \
  GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 \
  filesrc location=input.bin ! \
  s2sink basin=my-basin stream=media
```

Cloud source example:

```sh
S2_ACCESS_TOKEN='...' \
  GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 \
  s2src basin=my-basin stream=media \
    start-mode=sequence start-seq-num=0 ! filesink location=output.bin
```

For S2 Lite, set both endpoint properties to the same HTTP endpoint:

```sh
GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 \
  fakesrc num-buffers=1 filltype=zero sizetype=fixed sizemax=16 ! \
  s2sink basin=test-basin stream=test-stream \
  access-token-file=/run/secrets/s2-token \
  account-endpoint=http://127.0.0.1:8080 \
  basin-endpoint=http://127.0.0.1:8080
```

The two endpoints must be configured together, use the same scheme, and
contain no user-info. HTTP is accepted without additional configuration only
for loopback S2 Lite endpoints (`localhost`, IPv4 loopback, or `::1`). Remote
plaintext HTTP endpoints are rejected by default because they expose
credentials and data. Set `allow-insecure-endpoints=true` only for controlled
test networks; it is strongly discouraged elsewhere. TLS certificate
verification cannot be disabled.

## Record metadata

`s2src` attaches a custom `GstS2RecordMeta` to every output buffer. All six
top-level fields are required; a field with the wrong GValue type makes the
metadata invalid.

| Field | GStreamer value type | Required | Meaning from `s2src` | `s2sink` behavior |
|---|---|---:|---|---|
| `basin` | `String` (`G_TYPE_STRING`) | yes | Configured source basin name | Validates the field, but does not use it for routing |
| `stream` | `String` (`G_TYPE_STRING`) | yes | Configured source stream name | Validates the field, but does not use it for routing |
| `seq-num` | `u64` (`G_TYPE_UINT64`) | yes | Sequence number assigned by S2 | Validates only; it is not an append precondition |
| `timestamp` | `u64` (`G_TYPE_UINT64`) | yes | Record timestamp. Service-assigned values are Unix-epoch milliseconds; user-specified values are preserved as-is and may use an application-defined domain | Copies it to the appended record only when `preserve-timestamp=true`; otherwise validates only |
| `is-command` | `bool` (`G_TYPE_BOOLEAN`) | yes | Whether this is an S2 command record | Validates the command representation and rejects command metadata |
| `headers` | ordered `gst::Array` (`GST_TYPE_ARRAY`) | yes | All record headers in original order; an empty array is valid | Validates and appends regular-record headers in the same order |

Each `headers` array entry is a `gst::Structure`
(`GST_TYPE_STRUCTURE`) named `s2-header`:

| Field | GStreamer value type | Required | Rules |
|---|---|---:|---|
| `name` | `glib::Bytes` (`G_TYPE_BYTES`) | yes | Arbitrary binary bytes. Empty names are invalid for regular records |
| `value` | `glib::Bytes` (`G_TYPE_BYTES`) | yes | Arbitrary binary bytes, including an empty value |

Header order and duplicate names are significant and preserved. Names and
values need not be UTF-8. A regular record has `is-command=false` and no empty
header names. A command record has `is-command=true` and exactly one
`s2-header` whose `name` is empty; its binary `value` is the command payload.
Any disagreement between `is-command` and that header shape is invalid.
`s2sink` rejects command records even when the shape is valid, so a media
pipeline cannot fabricate or replay S2 fence or trim commands.

The following language-neutral shape uses inert placeholders. It describes
types and nesting, not a textual parser syntax:

```text
GstS2RecordMeta {
  basin: String = "<source-basin>",
  stream: String = "<source-stream>",
  seq-num: u64 = <sequence-number>,
  timestamp: u64 = <record-timestamp>,
  is-command: bool = false,
  headers: Array = [
    s2-header {
      name: Bytes = <binary-name>,
      value: Bytes = <binary-value>
    }
  ]
}
```

When `GstS2RecordMeta` is present, `s2sink` validates the complete structure
even if `preserve-timestamp=false`. Without metadata the sink provides no
headers or client timestamp; timestamp handling follows the stream's S2
timestamping configuration. Basin and stream routing, the fencing token, and
the initial sequence precondition always come from element properties, never
buffer metadata.

## Delivery and resume semantics

A successful sink `render` means that the local bounded queue accepted the
record. Successful EOS or normal stop means every preceding accepted record
was acknowledged by S2. The default `append-retry-policy=no-side-effects`
fails ambiguous appends instead of risking duplicates. Setting it to `all`
allows retries that may duplicate records.

The source has no built-in persisted checkpoint: returning a buffer does not
prove downstream processing is durable. A full stop or process restart reuses
the configured start position and can replay records. Applications needing a
durable cursor should persist `seq-num + 1` only after their own processing
acknowledgement, then restart with `start-mode=sequence` and that value.

These elements do not provide exactly-once processing. Within an uninterrupted
read session, records are emitted in S2 order; reconnect and resume behavior
follows the official SDK.

## Property reference

The tables below cover the properties defined by this plugin; inherited
`GstBaseSrc` and `GstBaseSink` properties are documented by GStreamer. Every
plugin property is readable and writable only while the element is in `NULL`
or `READY`. Numeric timeout and delay values are nanoseconds unless a row says
otherwise. Counts and byte limits are not time values.

### Shared source and sink properties

| Property | Default | Units and range | Mutable | Operational effect |
|---|---:|---|---|---|
| `basin` | unset (`null`) | Valid S2 basin name | `NULL`/`READY` | Required. Selects the single basin used by the element |
| `stream` | unset (`null`) | Valid S2 stream name | `NULL`/`READY` | Required. Selects the single stream used by the element |
| `access-token-file` | unset (`null`) | File path | `NULL`/`READY` | Reads the S2 access token from the file at startup. When unset, uses `S2_ACCESS_TOKEN`; the file takes precedence |
| `account-endpoint` | unset (`null`) | Endpoint string | `NULL`/`READY` | Overrides the SDK account endpoint. Must be set together with `basin-endpoint` |
| `basin-endpoint` | unset (`null`) | Endpoint string | `NULL`/`READY` | Overrides the SDK basin endpoint. Must be set together with `account-endpoint` |
| `allow-insecure-endpoints` | `false` | Boolean | `NULL`/`READY` | Allows remote plaintext HTTP only when `true`; it does not disable TLS certificate verification |
| `connection-timeout` | `3000000000` (3 s) | Nanoseconds, `1..=u64::MAX` | `NULL`/`READY` | Bounds S2 connection establishment |
| `request-timeout` | `5000000000` (5 s) | Nanoseconds, `1..=u64::MAX` | `NULL`/`READY` | Bounds each S2 request |
| `retry-max-attempts` | `3` | Total attempts, `1..=u32::MAX` | `NULL`/`READY` | Includes the initial request attempt |
| `retry-min-delay` | `100000000` (100 ms) | Nanoseconds, `0..=u64::MAX` | `NULL`/`READY` | Minimum SDK retry base delay |
| `retry-max-delay` | `1000000000` (1 s) | Nanoseconds, `0..=u64::MAX` | `NULL`/`READY` | Maximum SDK retry base delay; must be at least `retry-min-delay` |
| `compression` | `none` | Enum: `none`, `gzip`, `zstd` | `NULL`/`READY` | Selects SDK request and response compression |
| `queue-capacity` | `64` | Records, `1..=u32::MAX` | `NULL`/`READY` | Bounds records between GStreamer and the worker. Source reads back-pressure at this count; sink `render` waits when its accepted-record queue is full |

Explicit account and basin endpoints are a pair: setting only one is invalid,
and their schemes must match. Endpoints cannot contain user-info. HTTPS is
accepted; HTTP is accepted without opt-in only for S2 Lite on `localhost`, an
IPv4 loopback address, or `::1`. Remote plaintext HTTP requires
`allow-insecure-endpoints=true` and exposes credentials and record data, so it
should be limited to controlled test networks.

### Source-only properties

| Property | Default | Units and range | Mutable | Operational effect |
|---|---:|---|---|---|
| `caps` | unset (`null`) | GStreamer caps | `NULL`/`READY` | Optionally fixes the caps placed on source output; otherwise the pad remains `ANY` |
| `start-mode` | `earliest` | Enum: `earliest`, `sequence`, `timestamp`, `tail-offset` | `NULL`/`READY` | Selects which start field is used. `earliest` reads from sequence 0 |
| `start-seq-num` | `0` | Sequence number, `0..=u64::MAX` | `NULL`/`READY` | Used only by `start-mode=sequence`; starts at that sequence number |
| `start-timestamp` | `0` | Record timestamp, `0..=u64::MAX` | `NULL`/`READY` | Used only by `start-mode=timestamp`. Service-assigned timestamps are Unix-epoch milliseconds, but user-specified stream timestamps are passed through as-is |
| `tail-offset` | `0` | Records before tail, `0..=u64::MAX` | `NULL`/`READY` | Used only by `start-mode=tail-offset`; zero starts at the current tail |
| `clamp-to-tail` | `false` | Boolean | `NULL`/`READY` | When `true`, asks S2 to clamp an unwritten start position to the current tail. When `false`, a sequence or timestamp beyond the current tail makes the read fail before records are emitted |
| `ignore-command-records` | `false` | Boolean | `NULL`/`READY` | Asks S2 to omit command records from the read session |

Only the field selected by `start-mode` affects a new read session; the other
three stored start values are ignored. A flushing interruption within the
same started session resumes at the last delivered `seq-num + 1`, but a full
stop or process restart reuses these configured properties as described under
delivery and resume semantics.

### Sink-only properties

| Property | Default | Units and range | Mutable | Operational effect |
|---|---:|---|---|---|
| `batch-linger` | `5000000` (5 ms) | Nanoseconds, `0..=u64::MAX` | `NULL`/`READY` | Maximum producer wait for accumulating an append batch |
| `batch-max-records` | `1000` | Records, `1..=1000` | `NULL`/`READY` | Maximum record count in one S2 append batch |
| `batch-max-bytes` | `1048576` (1 MiB) | Metered bytes, `8..=1048576` | `NULL`/`READY` | Maximum metered size of one S2 append batch |
| `max-unacked-bytes` | `5242880` (5 MiB) | Metered bytes, `1048576..=u32::MAX` | `NULL`/`READY` | Bounds producer data submitted but not yet acknowledged by S2 |
| `append-retry-policy` | `no-side-effects` | Enum: `no-side-effects`, `all` | `NULL`/`READY` | `no-side-effects` fails ambiguous appends; `all` may retry them and can create duplicate records |
| `fencing-token-file` | unset (`null`) | File path | `NULL`/`READY` | Loads an optional S2 fencing-token append precondition at startup |
| `match-seq-num-enabled` | `false` | Boolean | `NULL`/`READY` | Applies `match-seq-num` as an initial append precondition only when enabled |
| `match-seq-num` | `0` | Sequence number, `0..=u64::MAX` | `NULL`/`READY` | Expected initial stream sequence; ignored when `match-seq-num-enabled=false` |
| `preserve-timestamp` | `false` | Boolean | `NULL`/`READY` | Copies the metadata timestamp unchanged into the append record. Service-assigned timestamps use Unix-epoch milliseconds, while user-specified values retain their own domain |
| `shutdown-timeout` | `10000000000` (10 s) | Nanoseconds, `1..=u64::MAX` | `NULL`/`READY` | Maximum wait for accepted records to be acknowledged during EOS or a normal state stop |

The fencing token and enabled match-sequence condition are both configured on
the producer and may apply together; metadata never supplies either
precondition. A failed precondition terminates the sink worker. A successful
`render` confirms only admission to the `queue-capacity` queue. Successful EOS
or normal stop closes the producer and waits for all submission tickets;
timeout or worker failure reports an error, and accepted records may remain
unconfirmed.

This design was informed by the public S2 behavior in Bento. No Bento code was
copied; its application-level batch acknowledgement and multi-stream cache
model were deliberately translated into native, single-pad GStreamer
semantics.

Licensed under the Apache License 2.0.
