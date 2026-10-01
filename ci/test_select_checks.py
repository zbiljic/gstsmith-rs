"""Run with python3 ci/test_select_checks.py; uses an isolated Git repository."""

import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

from select_checks import CRATES, SCOPES, select_checks


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    manifest = Path(__file__).resolve().parents[1] / "Cargo.toml"
    members = set(tomllib.loads(manifest.read_text())["workspace"]["members"])
    classified = set(CRATES)
    assert members == classified, (
        "Update GSTREAMER_124_CRATES / GSTREAMER_128_CRATES in ci/select_checks.py: "
        f"unclassified={sorted(members - classified)}, removed={sorted(classified - members)}"
    )
    assert len(CRATES) == len(classified), "Each crate must have exactly one GStreamer classification"

    cases = {
        "net/nats/src/lib.rs": "rust format packaging nats gstreamer124",
        "net/s2/tests/s2_lite.rs": "rust format s2 gstreamer124",
        "analytics/vlm/src/lib.rs": "rust format packaging gstreamer124",
        "analytics/tract-inference/src/lib.rs": "rust format packaging tract ort",
        "analytics/ort-inference/src/lib.rs": "rust format packaging ort",
        "analytics/nanodet/src/lib.rs": "rust format packaging",
        "analytics/inference-common/tests/fixtures/model.onnx": "rust tract ort",
        "analytics/tract-inference/tests/fixtures/model.onnx": "rust tract ort",
        ".config/mise/tasks/test/integration/nats": "nats",
        ".config/mise/tasks/test/integration/s2": "s2",
        "deny.toml": "deps",
        "rustfmt.toml": "format",
        "Cargo.lock": " ".join(SCOPES),
        "generic/console/Cargo.toml": " ".join(SCOPES),
        ".config/mise/conf.d/rust.toml": " ".join(SCOPES),
        ".github/workflows/ci.yml": " ".join(SCOPES),
        "ci/select_checks.py": " ".join(SCOPES),
        "README.md": "",
        "net/nats/README.md": "",
        ".github/renovate.json": "",
    }
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci@example.invalid")
            for filename in cases:
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original\n")
            git("add", ".")
            git("commit", "-qm", "baseline")
            base = git("rev-parse", "HEAD")
            for filename, expected in cases.items():
                # Reset only this disposable test repository between cases.
                git("reset", "--hard", base)
                Path(filename).unlink()
                git("add", "-u")
                git("commit", "-qm", "delete file")
                checks = select_checks(base)
                actual = {name for name in SCOPES if checks[name]}
                assert actual == set(expected.split()), (filename, checks)
                assert checks["any"] == bool(actual)
            git("reset", "--hard", base)
            git("mv", "net/nats/src/lib.rs", "net/s2/src.rs")
            git("commit", "-qm", "move between plugins")
            checks = select_checks(base)
            assert checks["nats"] and checks["s2"]
            assert all(select_checks("").values())
            assert all(select_checks("0" * 40).values())
            assert not any(select_checks("HEAD").values())
        finally:
            os.chdir(original)
    print("CI change selection checks passed")


if __name__ == "__main__":
    main()
