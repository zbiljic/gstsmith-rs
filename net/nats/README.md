# Core NATS GStreamer plugin

This crate provides `natssrc` and `natssink`, reusable Core NATS transports for
arbitrary bytes. Each NATS message maps to exactly one GStreamer buffer,
including an empty message. Both pads advertise `ANY`; `natssrc` can apply
fixed out-of-band caps with its `caps` property.

Core NATS is at-most-once and temporally coupled: a message sent while no
subscriber is active is not retained. Queue groups load-balance each message
across active members of the same group. JetStream persistence,
acknowledgements, redelivery, and automatic request/reply responses are
deliberately outside this plugin.

## Integration tests

The normal `mise run pre-commit` gate is offline/local and does not start a
broker. To run this crate's ignored integration suite, start a reachable Core
NATS server and run:

```sh
mise run test:integration:nats
```

The task defaults `NATS_TEST_URL` to `nats://127.0.0.1:4222`; set that
environment variable before invoking it to test another endpoint.

## Examples

These examples assume `mise run build` has completed from the repository root.

Publish bytes:

```sh
printf 'hello' | GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 fdsrc ! \
  natssink servers=nats://127.0.0.1:4222 subject=demo.bytes
```

Subscribe to bytes:

```sh
GST_PLUGIN_PATH="$PWD/target/debug" gst-launch-1.0 \
  natssrc servers=nats://127.0.0.1:4222 \
  subject=demo.bytes caps=application/octet-stream ! fdsink
```

For availability-first operation, enable bounded dropping and initial
connection retries:

```text
natssink servers=nats://127.0.0.1:4222 subject=<subject> \
  queue-capacity=64 drop-on-full=true retry-on-initial-connect=true \
  drain-timeout=2000000000
```

## Shared connection properties

All configurable properties are mutable only through READY.

| Property | Default | Meaning |
|---|---:|---|
| `servers` | `nats://127.0.0.1:4222` | Comma-separated server URLs without user-info |
| `connection-name` | unset | Monitoring name; the element name is the fallback |
| `credentials-file` | unset | User-credentials file containing JWT and seed |
| `nkey-file` | unset | File containing an NKey seed |
| `tls-required` | `false` | Require TLS |
| `tls-ca-file` | unset | PEM root CA bundle |
| `tls-client-cert-file` | unset | PEM client certificate chain |
| `tls-client-key-file` | unset | PEM client private key |
| `connection-timeout` | `5000000000` | Initial connection timeout, nanoseconds |
| `max-reconnects` | `0` | Consecutive reconnect limit; zero is unlimited |
| `retry-on-initial-connect` | `false` | Connect in the background after initial failure |

`credentials-file` and `nkey-file` are mutually exclusive. Client certificate
and key files must be configured together. Raw JWTs, seeds, passwords, and
tokens are intentionally not exposed as readable GObject properties. Server
URLs must not contain user-info; configure authentication with
`credentials-file` or `nkey-file` instead.

For a CA bundle:

```text
natssrc subject=events tls-required=true tls-ca-file=/run/secrets/nats-ca.pem
```

For mutual TLS:

```text
natssink subject=events tls-required=true \
  tls-client-cert-file=/run/secrets/client-cert.pem \
  tls-client-key-file=/run/secrets/client-key.pem
```

## Source properties

| Property | Default | Meaning |
|---|---:|---|
| `subject` | empty (required) | Subscription subject; wildcards are allowed |
| `queue-group` | empty | Optional load-balancing queue group |
| `subscription-capacity` | `1024` | Pending client subscription messages |
| `caps` | unset | Optional fixed output caps |

`natssrc` is live, non-seekable, uses time format, and timestamps output with
pipeline running time.

## Sink properties

| Property | Default | Meaning |
|---|---:|---|
| `subject` | empty | Fixed subject, or use message metadata |
| `headers` | empty | Array of fixed `nats-header` structures |
| `queue-capacity` | `64` | Messages awaiting the async publisher |
| `drop-on-full` | `false` | Drop newest instead of failing on overflow |
| `drain-timeout` | `2000000000` | Stop-time drain/flush limit, nanoseconds |
| `dropped-messages` | `0` | Read-only current-run overflow counter |

The queue capacity bounds message count, not payload bytes. Applications with
very large buffers should choose a correspondingly small queue. Overflow is an
error by default. With `drop-on-full=true`, the newest buffer is dropped and
`dropped-messages` is incremented.

Each `headers` entry is a `nats-header` structure with string `name` and
`value` fields, matching the message metadata format below. Repeated names are
preserved. Fixed headers are published first, followed by per-buffer metadata
headers, so both values remain available when a name appears in both places.
Malformed entries fail element startup before the broker connection is made.

## Message envelope metadata

`natssrc` attaches simple custom meta named `GstNatsMessageMeta`:

- `subject`: required actual NATS subject;
- `reply-subject`: optional non-empty reply subject;
- `headers`: optional array of `nats-header` structures, each containing
  string `name` and `value` fields. Repeated names remain separate entries.

`natssink` passes the reply subject and headers through, appending them after
its fixed `headers`. A non-empty sink `subject` overrides only the metadata
subject. With an empty sink subject, valid metadata is required. The metadata
is an envelope bridge only; it does not make the source a request/reply
responder.
