#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Capture one fail-closed live-process snapshot from a Vast outer container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0, "snapshot must run as root")
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is not None,
        "expected commit is invalid",
    )
    for path in (args.pid_file, args.output.parent):
        require(path.is_absolute(), f"{path} must be absolute")
    require(
        args.pid_file.is_file()
        and not args.pid_file.is_symlink()
        and args.pid_file.resolve(strict=True) == args.pid_file,
        "PID file is invalid",
    )
    require(
        stat.S_IMODE(args.pid_file.stat().st_mode) == 0o600,
        "PID file must have mode 0600",
    )
    require(not args.output.exists() and not args.output.is_symlink(), "output exists")
    process_id = int(args.pid_file.read_text().strip())
    require(process_id > 1, "server PID is invalid")
    status = {}
    for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    require(status.get("Uid", "").split()[:1] == ["2000"], "server UID drifted")
    require(status.get("Gid", "").split()[:1] == ["0"], "server GID drifted")
    executable = Path(os.readlink(f"/proc/{process_id}/exe"))
    command = Path(f"/proc/{process_id}/cmdline").read_bytes().split(b"\0")
    process_stat = Path(f"/proc/{process_id}/stat").read_text().split()
    require(len(process_stat) > 21 and command and command[0], "process is malformed")
    start_ticks = int(process_stat[21])
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    gpu_line = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    ).stdout.strip()
    require(gpu_line and "\n" not in gpu_line, "snapshot must see exactly one GPU")
    name, gpu_uuid, memory, driver, capability = (
        field.strip() for field in gpu_line.split(",")
    )
    require(name == "NVIDIA GeForce RTX 5070 Ti", "wrong GPU")
    require(capability == "12.0", "wrong compute capability")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "pid": process_id,
        "uid": 2000,
        "gid": 0,
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "command_sha256": hashlib.sha256(b"\0".join(command)).hexdigest(),
        "process_start_ticks": start_ticks,
        "process_started_at_unix": boot_time + start_ticks / os.sysconf("SC_CLK_TCK"),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "machine_id_sha256": hashlib.sha256(
            Path("/etc/machine-id").read_text().strip().encode()
        ).hexdigest(),
        "gpu": {
            "name": name,
            "uuid": gpu_uuid,
            "memory_total_mib": int(memory),
            "driver_version": driver,
            "compute_capability": [12, 0],
        },
        "fork_commit": args.expected_commit,
        "wheel_version": f"0.28.0+verse.{args.expected_commit[:12]}",
        "model_revision": "e2c6cd9c3302e91c032a378a607009c82ba16fac",
    }
    args.output.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "pass", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
