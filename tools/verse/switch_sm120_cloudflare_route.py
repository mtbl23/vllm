#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Inspect or compare-and-swap the stable Verse Free tunnel CNAME.

The command changes one exact proxied CNAME record from one exact Cloudflare
Tunnel UUID to another. It never creates, deletes, searches for, or selects a
resource by name. Production use is intentionally gated by ``--apply``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ROOT = "https://api.cloudflare.com/client/v4"
HEX_ID = re.compile(r"^[0-9a-f]{32}$")
TUNNEL_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
PROXY_ENV = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
)


def _read_secret(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Cloudflare API token path must be an absolute regular file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("Cloudflare API token file must be owner-only")
    if path.stat().st_uid != os.geteuid():
        raise ValueError("Cloudflare API token file must be owned by the caller")
    raw = path.read_bytes()
    if len(raw) > 4096 or b"\r" in raw or b"\0" in raw:
        raise ValueError("Cloudflare API token file has invalid bytes")
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("Cloudflare API token file must contain one non-empty line")
    return lines[0].decode("utf-8")


def _tunnel_cname(tunnel_id: str) -> str:
    tunnel_id = tunnel_id.strip().lower()
    if not TUNNEL_ID.fullmatch(tunnel_id):
        raise ValueError("tunnel ID must be an exact lowercase UUID")
    return f"{tunnel_id}.cfargotunnel.com"


def _api_request(
    *, method: str, path: str, token: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        API_ROOT + path,
        data=body,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read(1_000_001)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Cloudflare API returned HTTP {error.code}") from error
    if len(raw) > 1_000_000:
        raise RuntimeError("Cloudflare API response exceeded the safety limit")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or parsed.get("success") is not True:
        raise RuntimeError("Cloudflare API returned an unsuccessful response")
    result = parsed.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Cloudflare API response is missing an object result")
    return result


def _validate_record(
    record: dict[str, Any], *, record_id: str, hostname: str, expected_content: str
) -> None:
    if record.get("id") != record_id:
        raise RuntimeError("Cloudflare readback returned a different DNS record ID")
    if record.get("type") != "CNAME" or record.get("name") != hostname:
        raise RuntimeError("the selected record is not the exact Verse Free CNAME")
    if record.get("proxied") is not True:
        raise RuntimeError("the Verse Free CNAME must remain Cloudflare-proxied")
    if str(record.get("content", "")).rstrip(".").lower() != expected_content:
        raise RuntimeError(
            "the current tunnel target does not match the expected target"
        )


def _reserve_receipt(path: Path, payload: dict[str, Any]) -> int:
    if not path.is_absolute() or path.exists():
        raise ValueError("receipt must be a new absolute path")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    os.write(fd, encoded)
    os.fsync(fd)
    return fd


def _finish_receipt(fd: int, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, encoded)
    os.fsync(fd)


def _acquire_lock(path: Path) -> int:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("cutover lock must be an absolute non-symlink path")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
        os.close(fd)
        raise ValueError("cutover lock must have exact mode 0600")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(fd)
        raise RuntimeError(
            "another cutover operation holds the deployment lock"
        ) from error
    return fd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "switch"))
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--expected-current-tunnel", required=True)
    parser.add_argument("--target-tunnel")
    parser.add_argument("--api-token-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not HEX_ID.fullmatch(args.zone_id) or not HEX_ID.fullmatch(args.record_id):
        parser.error("zone ID and record ID must be exact 32-character lowercase IDs")
    if not re.fullmatch(r"[a-z0-9.-]+", args.hostname) or "." not in args.hostname:
        parser.error("hostname must be an exact lowercase DNS name")
    polluted = [name for name in PROXY_ENV if os.environ.get(name)]
    if polluted:
        parser.error(
            "proxy environment overrides are forbidden: " + ", ".join(polluted)
        )

    current = _tunnel_cname(args.expected_current_tunnel)
    token = _read_secret(args.api_token_file)
    path = f"/zones/{args.zone_id}/dns_records/{args.record_id}"
    record = _api_request(method="GET", path=path, token=token)
    _validate_record(
        record,
        record_id=args.record_id,
        hostname=args.hostname,
        expected_content=current,
    )

    if args.action == "inspect":
        print(
            json.dumps(
                {
                    "status": "match",
                    "record_id": args.record_id,
                    "hostname": args.hostname,
                    "tunnel": args.expected_current_tunnel,
                    "modified_on": record.get("modified_on"),
                },
                sort_keys=True,
            )
        )
        return 0

    if (
        not args.apply
        or not args.target_tunnel
        or args.receipt is None
        or args.lock is None
    ):
        parser.error("switch requires --apply, --target-tunnel, --receipt, and --lock")
    target = _tunnel_cname(args.target_tunnel)
    if target == current:
        parser.error("current and target tunnels must differ")

    pending = {
        "schema_version": 1,
        "status": "pending",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "zone_id": args.zone_id,
        "record_id": args.record_id,
        "hostname": args.hostname,
        "from_tunnel": args.expected_current_tunnel,
        "to_tunnel": args.target_tunnel,
        "expected_modified_on": record.get("modified_on"),
    }
    lock_fd = _acquire_lock(args.lock)
    receipt_fd: int | None = None
    try:
        receipt_fd = _reserve_receipt(args.receipt, pending)
        latest = _api_request(method="GET", path=path, token=token)
        _validate_record(
            latest,
            record_id=args.record_id,
            hostname=args.hostname,
            expected_content=current,
        )
        if latest.get("modified_on") != record.get("modified_on"):
            raise RuntimeError("DNS record changed after preflight; refusing mutation")
        changed = _api_request(
            method="PATCH",
            path=path,
            token=token,
            payload={"content": target},
        )
        _validate_record(
            changed,
            record_id=args.record_id,
            hostname=args.hostname,
            expected_content=target,
        )
        readback = _api_request(method="GET", path=path, token=token)
        _validate_record(
            readback,
            record_id=args.record_id,
            hostname=args.hostname,
            expected_content=target,
        )
        receipt = {
            **pending,
            "status": "verified",
            "changed_at": datetime.now(timezone.utc).isoformat(),
            "modified_on": readback.get("modified_on"),
        }
        _finish_receipt(receipt_fd, receipt)
    finally:
        if receipt_fd is not None:
            os.close(receipt_fd)
        os.close(lock_fd)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
