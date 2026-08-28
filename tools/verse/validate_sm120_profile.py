#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

EXPECTED_PROFILE = {
    "VERSE_PROFILE_VERSION": "2",
    "VERSE_SERVED_MODEL_NAME": "verse-free",
    "VERSE_MODEL_REPOSITORY": "marak12/verse-campaign22-gemma4-12b-compact",
    "VERSE_MODEL_REVISION": "e2c6cd9c3302e91c032a378a607009c82ba16fac",
    "VERSE_MODEL_SUBDIR": ("quantized/w4a4/candidate-unified-checkpoint-010674-nvfp4"),
    "VERSE_MODEL_MANIFEST_SHA256": (
        "159e4219ea3ffd636a5f7fa92ee4ecb0a05c194d8245b27605dc4725461c768d"
    ),
    "VERSE_MAX_MODEL_LEN": "6144",
    "VERSE_MAX_NUM_SEQS": "38",
    "VERSE_MAX_NUM_BATCHED_TOKENS": "256",
    "VERSE_GPU_MEMORY_UTILIZATION": "0.94",
    "VERSE_KV_CACHE_DTYPE": "nvfp4",
    "VERSE_KV_CACHE_LAYOUT": "HND",
    "VERSE_ATTENTION_BACKEND": "FLASHINFER",
    "VERSE_QUANTIZATION": "modelopt_fp4",
    "VERSE_LINEAR_BACKEND": "flashinfer_b12x",
    "VERSE_COMPILATION_CONFIG": '{"cudagraph_mode":"NONE"}',
}
HEX40_RE = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
PROFILE_LINE_RE = re.compile(r"([A-Z][A-Z0-9_]*)=(.*)")


def load_profile(path: Path) -> dict[str, str]:
    profile: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROFILE_LINE_RE.fullmatch(line)
        if match is None:
            raise SystemExit(f"invalid profile syntax on line {line_number}")
        key, raw_value = match.groups()
        if key in profile:
            raise SystemExit(f"duplicate profile key {key!r}")
        value = raw_value
        if value.startswith("'") or value.endswith("'"):
            if len(value) < 2 or not (value.startswith("'") and value.endswith("'")):
                raise SystemExit(f"unbalanced profile quotes on line {line_number}")
            value = value[1:-1]
            if "'" in value:
                raise SystemExit(f"embedded profile quote on line {line_number}")
        profile[key] = value
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable Verse SM120 serving profile."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("sm120_profile.env"),
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--emit-shell",
        action="store_true",
        help="Emit the validated profile as shell-safe assignments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_profile(args.profile)
    if profile != EXPECTED_PROFILE:
        raise SystemExit(
            "SM120 profile drifted from its validated constants:\n"
            + json.dumps(profile, indent=2, sort_keys=True)
        )
    if not HEX40_RE.fullmatch(args.expected_commit):
        raise SystemExit("expected fork commit must be a 40-character commit SHA")
    if not IMAGE_DIGEST_RE.fullmatch(args.image):
        raise SystemExit("production image must be pinned by sha256 digest")
    if args.emit_shell:
        for key, value in profile.items():
            print(f"{key}={shlex.quote(value)}")
    else:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "profile": profile,
                    "image": args.image,
                    "model": profile["VERSE_MODEL_REPOSITORY"],
                    "model_revision": profile["VERSE_MODEL_REVISION"],
                    "model_subdir": profile["VERSE_MODEL_SUBDIR"],
                    "fork_commit": args.expected_commit,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
