# VLM analysis

The `vlm` plugin provides `vlmanalysis`, an asynchronous analysis element for
JPEG streams. It sends selected frames to a non-streaming OpenAI-compatible Chat
Completions endpoint while passing every input buffer downstream unchanged,
including its payload, timestamps, offsets, flags, and metadata.

## Quick start

Use `videoconvert` and `jpegenc` when the source does not already produce
`image/jpeg`:

```sh
gst-launch-1.0 -m \
  videotestsrc \
  ! videoconvert \
  ! jpegenc \
      quality=85 \
  ! vlmanalysis \
      model=microsoft/Phi-3.5-vision-instruct \
      user-prompt="Describe the scene." \
  ! fakesink
```

The `-m` option prints the `vlmanalysis-result` and `vlmanalysis-error`
messages posted on the pipeline bus.

## Provider setup and security

Set `endpoint` to a complete OpenAI-compatible `/v1/chat/completions` URL and
set the required `model`. vLLM, Ollama, Gemini, and Amazon Bedrock are supported
only through OpenAI-compatible endpoints they expose; their native APIs are
not supported. The server operator is responsible for deploying a compatible
multimodal model and any server-side model chat template it requires.

`system-prompt` and `user-prompt` are sent as literal prompt content.

To send a Bearer credential, set `api-key-file` to a UTF-8 file. It is read
once when the element starts, trailing whitespace is removed, and empty
content is rejected. Redirects are disabled and HTTPS uses normal Web PKI
certificate validation. Plaintext HTTP is accepted for loopback addresses;
non-loopback HTTP requires `allow-insecure-http=true`, which exposes JPEG data
and credentials in transit.

## Sampling and lifecycle

`analysis-interval` defaults to five seconds and uses buffer PTS; zero selects
every buffer. Without PTS, only the first buffer while the sampler is empty is
selected unless the interval is zero. `frames-per-request` groups 1 through 10
selected frames in input order.

Each run starts at generation one. STREAM_START, SEGMENT, FLUSH_STOP, and a
backward PTS jump advance the generation and discard an incomplete batch.
Queued or active work may still finish with its older generation, which lets
consumers reject stale results. EOS does not submit or wait for an incomplete
batch.

Requests are processed in order. `queue-capacity` bounds complete batches
waiting for service. When the queue is full, the newest batch is dropped,
`dropped-batches` increases, and a `backpressure` message is posted without
blocking the stream. `drain-timeout` bounds shutdown; zero aborts immediately.

## Bus messages and counters

`vlmanalysis-result` contains:

- `request-id` (`u64`), `generation` (`u64`), `model` (string), `text` (string),
  `frame-count` (`u32`), and `latency` (`u64` nanoseconds).
- `start-pts` and `end-pts` (`GstClockTime`) when the batch contains valid PTS.
- `prompt-tokens` and `completion-tokens` (`u64`) when reported by the provider.

`vlmanalysis-error` contains `generation`, `kind`, a sanitized `message`, and
`frame-count`. It also contains `request-id` after batching and `http-status`
when an HTTP response supplied one. `kind` is `input`, `backpressure`,
`timeout`, `http`, or `response`. Invalid startup settings and an unavailable
worker are reported as normal GStreamer errors.

Messages and logs do not include response bodies, prompts, request JSON, image
data, authorization headers, credentials, or credential paths.

The read-only `submitted-requests`, `completed-requests`, `failed-requests`,
and `dropped-batches` counters describe the current run. Counters and request
IDs reset when the element starts.

## Resource limits

`max-frame-bytes` defaults to 8 MiB and `max-batch-bytes` to 16 MiB. Startup
requires `frames-per-request * max-frame-bytes <= max-batch-bytes`. Provider
responses are collected incrementally and rejected above 1 MiB.

At maximum settings, retained raw JPEG data can reach 192 MiB, plus about
86 MiB for the active request's encoded data URLs and JSON. Set
`max-frame-bytes`, `max-batch-bytes`, and `queue-capacity` for the available
memory budget.

## Best-effort OCR

For one transcription associated with one selected sample, set
`frames-per-request=1` and use this literal prompt:

> Transcribe all visible text as faithfully as possible. Preserve line breaks.
> If no text is visible, state that.

The result is unstructured model text. It does not guarantee exact recognition,
bounding boxes, confidence, language, reading order, layout, or freedom from
hallucinations, and there is no reliable machine-readable no-text sentinel.
With multiple frames, the result is undifferentiated batch-level text tied only
to the batch's start and end PTS. Use a dedicated OCR system when typed OCR
fields are required.

## Live smoke test

Local tests use loopback servers. To run the ignored test against a compatible
endpoint:

```sh
VLM_TEST_ENDPOINT="https://provider.example/v1/chat/completions" \
VLM_TEST_MODEL="model-name" \
VLM_TEST_API_KEY_FILE="/path/to/key" \
cargo test -p gst-plugin-vlm --test vlmanalysis live_openai_compatible_smoke \
  -- --ignored --exact
```

`VLM_TEST_API_KEY_FILE` is optional. The test sends a generated one-pixel JPEG;
do not use an endpoint where that request is unwanted or billable.
