# GStreamer lines plugin

The `lines` plugin provides bounded delimiter framing for arbitrary byte
streams. GStreamer source buffers are stream chunks and do not necessarily
correspond to logical records.

## Elements

- `lineparse`: Split stream chunks into record buffers and remove the
  configured delimiter.
- `lineenc`: Ensure every record buffer ends with the configured delimiter.

Use `lineparse` when downstream elements require one delimiter-framed record
per buffer. Use `lineenc` when a byte-stream sink requires a delimiter after
each input record.

## Examples

Run these pipelines from the repository root after `mise run build`.

```sh
# Frame newline-delimited stdin records.
printf 'one\ntwo\n' | \
  GST_PLUGIN_PATH="$PWD/target/debug" \
  gst-launch-1.0 -q \
    fdsrc \
    ! application/octet-stream \
    ! lineparse \
    ! fakesink

# Frame records, restore their delimiters, and write them to standard output.
printf 'one\ntwo' | \
  GST_PLUGIN_PATH="$PWD/target/debug" \
  gst-launch-1.0 -q \
    fdsrc \
    ! application/octet-stream \
    ! lineparse \
    ! lineenc \
    ! fdsink fd=1
```

Inspect the plugin and elements with:

```sh
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 lines
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 lineparse
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 lineenc
```

## Development

```sh
cargo check -p gst-plugin-lines --all-targets
cargo test -p gst-plugin-lines --all-targets
cargo clippy -p gst-plugin-lines --all-targets -- -D warnings
```
