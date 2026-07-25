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

Install the pinned toolchain and build every plugin in the workspace:

```sh
mise install
mise run build
```

Inspect any built plugin or element by name:

```sh
GST_PLUGIN_PATH="$PWD/target/debug" gst-inspect-1.0 <plugin-or-element>
```

The workspace build automatically includes new members, so the root README
does not need another build command when a plugin is added.

## Development

Run the complete workspace validation before submitting a change:

```sh
mise run pre-commit
```

For a faster iteration loop, target one plugin by its Cargo package name:

```sh
cargo check -p gst-plugin-<name> --all-targets
cargo test -p gst-plugin-<name> --all-targets
cargo clippy -p gst-plugin-<name> --all-targets -- -D warnings
```

These workspace and package-scoped commands are shared by all plugins; keep
plugin-specific behavior and pipeline examples with the relevant plugin
documentation rather than adding per-plugin build and test lists here.

## License

Licensed under the [Apache License 2.0](LICENSE).
