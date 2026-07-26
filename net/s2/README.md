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

`s2src` attaches `GstS2RecordMeta` to every buffer. It carries the configured
basin and stream, the service-assigned `seq-num` and `timestamp`, command
status, and an ordered array of binary name/value headers. Duplicate and
non-UTF-8 headers survive normal buffer copies.

`s2sink` reads regular-record headers from this metadata. Routing and append
preconditions always come from element properties, never buffer metadata.
Command metadata and empty header names are rejected so a media pipeline
cannot fabricate or replay S2 fence or trim commands.

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

## Important properties

Shared properties include `basin`, `stream`, `access-token-file`,
`account-endpoint`, `basin-endpoint`, `allow-insecure-endpoints`,
connection/request timeouts, retry attempts and delays, `compression`, and
`queue-capacity`.

Source start controls are `start-mode`, `start-seq-num`, `start-timestamp`,
`tail-offset`, `clamp-to-tail`, and `ignore-command-records`. `caps` optionally
sets fixed output caps.

Sink batching controls are `batch-linger`, `batch-max-records`,
`batch-max-bytes`, and `max-unacked-bytes`. Durability and append controls are
`append-retry-policy`, `fencing-token-file`, `match-seq-num-enabled`,
`match-seq-num`, `preserve-timestamp`, and `shutdown-timeout`.

This design was informed by the public S2 behavior in Bento. No Bento code was
copied; its application-level batch acknowledgement and multi-stream cache
model were deliberately translated into native, single-pad GStreamer
semantics.

Licensed under the Apache License 2.0.
