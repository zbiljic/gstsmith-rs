# gstsmith-rs

Repository containing [GStreamer](https://gstreamer.freedesktop.org/) plugins
and elements written in Rust.

## Plugins

- [`generic`](generic/)

  - [`console`](generic/console/): Console byte transports and a text-oriented
    debug tap.
    - `consolesrc`: Read raw byte chunks from standard input.
    - `consoleprint`: Print text-oriented buffers and pass them downstream
      unchanged.
    - `consolesink`: Write exact bytes to standard output or error.

- [`net`](net/)

  - [`nats`](net/nats/): Core NATS byte-message transports.
    - `natssrc`: Subscribe to Core NATS subjects as one buffer per message.
    - `natssink`: Publish one buffer per Core NATS message.
  - [`s2`](net/s2/): S2 durable-stream byte transports.
    - `s2src`: Read one configured S2 stream as one buffer per record.
    - `s2sink`: Append one buffer per S2 record with acknowledged shutdown.

- [`text`](text/)

  - [`lines`](text/lines/): Bounded delimiter framing for arbitrary byte
    streams.
    - `lineparse`: Split stream chunks into record buffers and remove the
      configured delimiter.
    - `lineenc`: Ensure every record buffer ends with the configured delimiter.

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

Run the complete offline/local validation gate before submitting a change:

```sh
mise run pre-commit
```

The gate above does not start external services. Service-dependent integration
suites are opt-in:

```sh
mise run test:integration
```

The composite task runs every integration suite. See each plugin's README for
its individual task, prerequisites, and environment configuration.

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
