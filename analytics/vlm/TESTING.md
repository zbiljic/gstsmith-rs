# Testing vlmanalysis

Run these commands from the repository root.

## Offline tests

```sh
mise exec -- cargo test -p gst-plugin-vlm --all-targets
```

These tests use local fixtures and loopback servers; the live test is ignored
unless explicitly requested.

## Live smoke test

Local tests use loopback servers. The ignored test sends one JPEG to a real
endpoint and checks for a completed, nonempty result. Both endpoint and model
are required when invoking it explicitly:

```sh
VLM_TEST_ENDPOINT="https://provider.example/v1/chat/completions" \
VLM_TEST_MODEL="model-name" \
VLM_TEST_IMAGE_FILE="/path/to/image.jpg" \
VLM_TEST_API_KEY_FILE="/path/to/key" \
mise exec -- cargo test -p gst-plugin-vlm --test vlmanalysis \
  live_openai_compatible_smoke -- --ignored --exact --nocapture
```

All other variables are optional:

| Variable | Default / behavior |
| --- | --- |
| `VLM_TEST_API_KEY_FILE` | No authentication; otherwise reads the credential file. |
| `VLM_TEST_ALLOW_INSECURE_HTTP` | `false`; explicitly set `true` for non-loopback HTTP. |
| `VLM_TEST_IMAGE_FILE` | Generated one-pixel JPEG; otherwise reads the supplied JPEG. |
| `VLM_TEST_USER_PROMPT` | Element default; otherwise sent literally. Ask for JSON in JSON modes. |
| `VLM_TEST_RESPONSE_FORMAT` | `default`; also accepts `text`, `json-object`, `json-schema`. |
| `VLM_TEST_RESPONSE_SCHEMA` | Unset; inline JSON Schema required for `json-schema`. |
| `VLM_TEST_REASONING_EFFORT` | Unset; sent literally when supplied. |
| `VLM_TEST_TOKEN_LIMIT_MODE` | `legacy`; also accepts `completion`. |
| `VLM_TEST_SAMPLING_MODE` | `configured`; also accepts `provider-default`. |
| `VLM_TEST_MAX_TOKENS` | `512`; positive integer. |
| `VLM_TEST_TIMEOUT_SECONDS` | `120`; positive integer request deadline, plus five seconds for the bus wait. |
| `VLM_TEST_EXPECT_JSON` | Unset; expected JSON object, validated before sending and compared by fields and values, ignoring whitespace and object key order. |
| `VLM_TEST_EXPECT_TEXT` | Unset; optionally assert a case-sensitive substring in the result. |

JSON modes check that the result is a JSON object. The test does not validate
schema adherence. `--nocapture` prints sanitized failure details or result
latency and token usage, without printing image data, credentials, or response
text. Only run against an endpoint where sending the image and any associated
usage charges are intended.

For a reproducible color check, generate a solid-red JPEG and configure your
endpoint and model. Set `VLM_TEST_API_KEY_FILE` if authentication is required,
and explicitly opt into `VLM_TEST_ALLOW_INSECURE_HTTP=true` for non-loopback HTTP:

```sh
vlm_test_dir=$(mktemp -d)
gst-launch-1.0 \
  videotestsrc num-buffers=1 pattern=red \
  ! video/x-raw,width=320,height=240 \
  ! videoconvert \
  ! jpegenc \
  ! filesink \
      location="$vlm_test_dir/red.jpg"

export VLM_TEST_ENDPOINT="https://provider.example/v1/chat/completions"
export VLM_TEST_MODEL="model-name"
export VLM_TEST_IMAGE_FILE="$vlm_test_dir/red.jpg"
export VLM_TEST_USER_PROMPT="Identify the dominant image color. Return only a raw JSON object with a color field and its lowercase string value. Do not use Markdown, code fences, backticks, commentary, or explanations."

VLM_TEST_TIMEOUT_SECONDS=300 \
VLM_TEST_RESPONSE_FORMAT=json-object \
VLM_TEST_EXPECT_JSON='{"color":"red"}' \
mise exec -- cargo test -p gst-plugin-vlm --test vlmanalysis \
  live_openai_compatible_smoke -- --ignored --exact --nocapture
```

Allow 300 seconds for the first request, then run warm checks separately with a
shorter deadline. Model loading is one possible reason for a slow first request;
a timeout alone proves neither loading nor lack of feature support. Repeat the
color check with `VLM_TEST_TIMEOUT_SECONDS=120` to measure a warm request.

To probe one schema constraint, add a required marker whose value appears only
in the schema, without changing the prompt above:

```sh
VLM_TEST_TIMEOUT_SECONDS=120 \
VLM_TEST_RESPONSE_FORMAT=json-schema \
VLM_TEST_RESPONSE_SCHEMA='{"type":"object","properties":{"color":{"type":"string"},"marker":{"type":"string","enum":["schema-check-041"]}},"required":["color","marker"],"additionalProperties":false}' \
VLM_TEST_EXPECT_JSON='{"color":"red","marker":"schema-check-041"}' \
mise exec -- cargo test -p gst-plugin-vlm --test vlmanalysis \
  live_openai_compatible_smoke -- --ignored --exact --nocapture
```

The expected object checks this marker constraint and the color result; passing
does not establish general schema enforcement. Exact JSON mismatches fail
without printing either object. Remove the temporary image directory afterward:

```sh
rm -r "$vlm_test_dir"
```
