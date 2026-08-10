# GStreamer console plugin

The `console` plugin provides standard-input/output byte transports and a
text-oriented debug tap. Framing, serialization, and media interpretation stay
in explicit upstream or downstream elements.

## Elements

- `consolesrc`: Read arbitrary byte chunks from standard input. Its source pad
  accepts any caps, and chunk boundaries are not record boundaries.
- `consolesink`: Write every input buffer exactly as received to standard
  output or error. Its sink pad accepts any caps and it does not add
  delimiters.
- `consoleprint`: Write supported text-oriented buffers to standard output or
  error, then pass the original buffer downstream unchanged.

`consoleprint` accepts `text/x-raw, format=utf8`, `application/json`, and
`application/x-json`. It validates that payload bytes are UTF-8, but it does
not parse or validate JSON syntax. The JSON caps describe accepted media
types; they do not make `consoleprint` a JSON codec.

## Composition

Run these pipelines from the repository root after `mise run build`.

Frame newline-delimited standard input explicitly:

```sh
printf 'one\ntwo\n' | \
  GST_PLUGIN_PATH="$PWD/target/debug" \
  gst-launch-1.0 -q \
    consolesrc \
    ! lineparse \
    ! fakesink
```

The byte transports also compose with optional payloaders, depayloaders,
encoders, and parsers without depending on them:

```text
... \
  ! gdppay \
  ! consolesink
consolesrc \
  ! gdpdepay \
  ! ...

... \
  ! jsongstenc \
  ! consolesink
consolesrc \
  ! jsongstparse \
  ! ...
```

`jsongstparse` expects the GStreamer envelope emitted by `jsongstenc`, not an
arbitrary JSON document.

Inspect the plugin and elements with:

```sh
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 console
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 consolesrc
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 consolesink
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 consoleprint
```

## Development

```sh
cargo check -p gst-plugin-console --all-targets
cargo test -p gst-plugin-console --all-targets
cargo clippy -p gst-plugin-console --all-targets -- -D warnings
```
