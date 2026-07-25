# gstsmith-rs

Repository containing [GStreamer](https://gstreamer.freedesktop.org/) plugins
and elements written in Rust.

## Plugins

- [`generic`](generic/)

  - [`console`](generic/console/): Console I/O for UTF-8 text and JSON buffers.
    - `consolesrc`: Read buffers from standard input.
    - `consoleprint`: Write buffers to standard output or error and pass them
      downstream.
    - `consolesink`: Write buffers to standard output or error.

## Building

```sh
cargo build -p gst-plugin-console
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 console
```

## Development

```sh
mise install
mise run pre-commit
```

## License

Licensed under the [Apache License 2.0](LICENSE).
