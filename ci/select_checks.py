"""Select CI checks with Git path filters; shared build inputs run everything."""

import json
import subprocess
import sys

SHARED = (
    ":(glob)**/Cargo.toml",
    "Cargo.lock",
    "mise.toml",
    "ruff.toml",
    "rust-toolchain",
    "rust-toolchain.toml",
    ".cargo",
    ":(glob).config/mise/**/*.toml",
    ".github/workflows",
    "ci",
)
GSTREAMER_124_CRATES = (
    "generic/console",
    "text/lines",
    "net/nats",
    "net/s2",
    "analytics/vlm",
    "analytics/ocrs",
    "utils/prometheus",
    "utils/statsd",
)
GSTREAMER_128_CRATES = (
    "analytics/inference-common",
    "analytics/nanodet",
    "analytics/ort-inference",
    "analytics/tract-inference",
)
CRATES = (*GSTREAMER_124_CRATES, *GSTREAMER_128_CRATES)
SCOPES = {
    "rust": CRATES,
    "format": (":(glob)**/*.rs", ":(glob)**/*.py", "rustfmt.toml", ".rustfmt.toml"),
    "deps": ("deny.toml",),
    "sort": (":(glob)**/Cargo.toml",),
    "packaging": (
        *CRATES,
        ":(glob,exclude)**/tests/**",
        ":(glob,exclude)**/benches/**",
        ":(glob,exclude)**/examples/**",
    ),
    "nats": ("net/nats", ".config/mise/tasks/test/integration/nats"),
    "s2": ("net/s2", ".config/mise/tasks/test/integration/s2"),
    "tract": ("analytics/tract-inference", "analytics/inference-common"),
    # ORT parity tests also use the Tract crate and its fixtures.
    "ort": (
        "analytics/ort-inference",
        "analytics/tract-inference",
        "analytics/inference-common",
    ),
    "gstreamer124": GSTREAMER_124_CRATES,
}


def changed(base, paths):
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--no-renames",
            base,
            "HEAD",
            "--",
            *paths,
            ":(glob,exclude)**/*.md",
        ],
        check=False,
    )
    if result.returncode not in (0, 1):
        result.check_returncode()
    return result.returncode == 1


def select_checks(base):
    # Manual runs, merge queues, and unavailable push bases get full coverage.
    full = (
        not base
        or subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
    )
    full = full or changed(base, SHARED)
    checks = {name: full or changed(base, paths) for name, paths in SCOPES.items()}
    checks["any"] = any(checks.values())
    return checks


if __name__ == "__main__":
    print(
        "checks=" + json.dumps(select_checks(sys.argv[1] if len(sys.argv) > 1 else ""))
    )
