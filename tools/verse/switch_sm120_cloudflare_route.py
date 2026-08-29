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
import hashlib
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from validate_sm120_gateway_release import load_owned_file, validate
from verify_sm120_public_gateway import _cloudflare_tunnel_ingress_binding, _NoRedirect

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


def _secure_parent(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("deployment state parent must be canonical and non-symlink")
    metadata = path.stat()
    if metadata.st_uid != os.geteuid():
        raise ValueError("deployment state parent must be owned by the caller")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("deployment state parent must not be group/world writable")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("deployment state ancestry must not contain symlinks")
        ancestor_metadata = ancestor.stat()
        if ancestor_metadata.st_uid not in (0, os.geteuid()):
            raise ValueError("deployment state ancestry has an untrusted owner")
        mode = stat.S_IMODE(ancestor_metadata.st_mode)
        writable = mode & (stat.S_IWGRP | stat.S_IWOTH)
        sticky_root = (
            ancestor != path
            and ancestor_metadata.st_uid == 0
            and bool(mode & stat.S_ISVTX)
            and bool(mode & stat.S_IWOTH)
        )
        if writable and not sticky_root:
            raise ValueError("deployment state ancestry is group/world writable")


def _load_public_proof(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("public proof must be an absolute regular non-symlink file")
    if path.resolve(strict=True) != path:
        raise ValueError("public proof path must be canonical")
    _secure_parent(path.parent)
    metadata = path.stat()
    if metadata.st_uid != os.geteuid():
        raise ValueError("public proof must be owned by the caller")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("public proof must not be group/world writable")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("public proof is not an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_target_binding(
    *,
    target_mode: str,
    manifest_path: Path | None,
    proof_path: Path,
    target_tunnel: str,
    proof_zone_id: str,
    proof_record_id: str,
    proof_hostname: str,
    proof_account_id: str,
    production_hostname: str,
    route_zone_id: str,
    route_record_id: str,
    expected_current_tunnel: str,
    expected_modified_on: str,
) -> dict[str, Any]:
    proof, proof_sha256 = _load_public_proof(proof_path)
    if (
        proof.get("status") != "pass"
        or proof.get("target_tunnel") != target_tunnel
        or proof.get("target_mode") != target_mode
    ):
        raise ValueError("public proof does not cover the exact target tunnel")
    dns_binding = proof.get("dns_binding")
    expected_dns_binding = {
        "zone_id": proof_zone_id,
        "record_id": proof_record_id,
        "hostname": proof_hostname,
        "record_type": "CNAME",
        "proxied": True,
        "content": _tunnel_cname(target_tunnel),
    }
    if not isinstance(dns_binding, dict) or any(
        dns_binding.get(name) != expected
        for name, expected in expected_dns_binding.items()
    ):
        raise ValueError(
            "public proof does not bind the exact DNS record to the tunnel"
        )
    tunnel_ingress = proof.get("tunnel_ingress_binding")
    expected_ingress = {
        "account_id": proof_account_id,
        "tunnel_id": target_tunnel,
        "proof_hostname": proof_hostname,
        "production_hostname": production_hostname,
        "service": "http://127.0.0.1:8080",
    }
    if not isinstance(tunnel_ingress, dict) or any(
        tunnel_ingress.get(name) != expected
        for name, expected in expected_ingress.items()
    ):
        raise ValueError("public proof does not bind production ingress to the tunnel")
    route_precondition = proof.get("route_precondition")
    expected_route_precondition = {
        "zone_id": route_zone_id,
        "record_id": route_record_id,
        "hostname": production_hostname,
        "record_type": "CNAME",
        "proxied": True,
        "content": _tunnel_cname(expected_current_tunnel),
        "modified_on": expected_modified_on,
    }
    if route_precondition != expected_route_precondition:
        raise ValueError("public proof does not bind the exact pre-cutover route state")
    identity: dict[str, str] = {}
    manifest_sha256: str | None = None
    if target_mode == "qualified-candidate":
        if manifest_path is None:
            raise ValueError("qualified candidate requires a release manifest")
        manifest, manifest_sha256 = load_owned_file(manifest_path)
        container = manifest.get("container")
        if not isinstance(container, dict):
            raise ValueError("target release manifest lacks container identity")
        identity = validate(
            manifest,
            container_id=str(container.get("id", "")),
            image_digest=str(container.get("image_digest", "")),
            fork_commit=str(manifest.get("source_commit", "")),
        )
        if proof.get("release_manifest_sha256") != manifest_sha256:
            raise ValueError("public proof does not cover the target release manifest")
        for name, expected in identity.items():
            if proof.get(name) != expected:
                raise ValueError(f"public proof target identity mismatch: {name}")
    elif target_mode == "recorded-rollback":
        if (
            manifest_path is not None
            or proof.get("release_manifest_sha256") is not None
        ):
            raise ValueError(
                "rollback proof must not claim a candidate release manifest"
            )
        if not isinstance(proof.get("endpoint"), str) or not isinstance(
            proof.get("model"), str
        ):
            raise ValueError("rollback proof lacks the recorded service identity")
        identity = {
            "endpoint": proof["endpoint"],
            "model": proof["model"],
        }
    else:
        raise ValueError("unsupported target mode")
    try:
        verified_at = datetime.fromisoformat(str(proof.get("verified_at", "")))
    except ValueError as exc:
        raise ValueError("public proof has an invalid verification time") from exc
    now = datetime.now(timezone.utc)
    if (
        verified_at.tzinfo is None
        or not now - timedelta(minutes=2) <= verified_at <= now
    ):
        raise ValueError("public proof is stale or from the future")
    return {
        **identity,
        "target_mode": target_mode,
        "target_tunnel": target_tunnel,
        "release_manifest_sha256": manifest_sha256,
        "public_proof_sha256": proof_sha256,
        "public_proof_verified_at": proof["verified_at"],
        "proof_dns_binding": dns_binding,
        "proof_tunnel_ingress_binding": tunnel_ingress,
        "proof_route_precondition": route_precondition,
    }


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
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
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
    _secure_parent(path.parent)
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
    _secure_parent(path.parent)
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
    parser.add_argument("--target-release-manifest", type=Path)
    parser.add_argument("--target-public-proof", type=Path)
    parser.add_argument("--target-proof-zone-id")
    parser.add_argument("--target-proof-record-id")
    parser.add_argument("--target-proof-hostname")
    parser.add_argument("--target-proof-account-id")
    parser.add_argument("--target-proof-api-token-file", type=Path)
    parser.add_argument(
        "--target-mode",
        choices=("qualified-candidate", "recorded-rollback"),
    )
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
        or args.target_public_proof is None
        or args.target_mode is None
        or args.target_proof_zone_id is None
        or args.target_proof_record_id is None
        or args.target_proof_hostname is None
        or args.target_proof_account_id is None
        or args.target_proof_api_token_file is None
    ):
        parser.error(
            "switch requires --apply, target tunnel/release/proof, receipt, and lock"
        )
    if (
        not HEX_ID.fullmatch(args.target_proof_account_id)
        or not HEX_ID.fullmatch(args.target_proof_zone_id)
        or not HEX_ID.fullmatch(args.target_proof_record_id)
        or not re.fullmatch(r"[a-z0-9.-]+", args.target_proof_hostname)
        or "." not in args.target_proof_hostname
    ):
        parser.error("target proof DNS identity is invalid")
    target = _tunnel_cname(args.target_tunnel)
    if target == current:
        parser.error("current and target tunnels must differ")
    target_binding = _validate_target_binding(
        manifest_path=args.target_release_manifest,
        proof_path=args.target_public_proof,
        target_tunnel=args.target_tunnel,
        target_mode=args.target_mode,
        proof_zone_id=args.target_proof_zone_id,
        proof_record_id=args.target_proof_record_id,
        proof_hostname=args.target_proof_hostname,
        proof_account_id=args.target_proof_account_id,
        production_hostname=args.hostname,
        route_zone_id=args.zone_id,
        route_record_id=args.record_id,
        expected_current_tunnel=args.expected_current_tunnel,
        expected_modified_on=str(record.get("modified_on", "")),
    )

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
        "target_binding": target_binding,
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
        current_ingress = _cloudflare_tunnel_ingress_binding(
            account_id=args.target_proof_account_id,
            target_tunnel=args.target_tunnel,
            proof_hostname=args.target_proof_hostname,
            production_hostname=args.hostname,
            token=_read_secret(args.target_proof_api_token_file),
        )
        if current_ingress != target_binding["proof_tunnel_ingress_binding"]:
            raise RuntimeError("target tunnel ingress changed after public proof")
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
