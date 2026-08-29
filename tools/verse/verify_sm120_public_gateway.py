#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Fail-closed public-route smoke test for the Verse SM120 gateway."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_sm120_gateway_release import load_owned_file, validate

MAX_RESPONSE_BYTES = 1_000_000
CF_API_ROOT = "https://api.cloudflare.com/client/v4"
HEX_ID = re.compile(r"[0-9a-f]{32}")
FORBIDDEN = (
    "/pause",
    "/abort_requests",
    "/invocations",
    "/metrics",
    "/docs",
    "/openapi.json",
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, url):
        return None


def _secret(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("credential path must be an absolute regular file")
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("credential files must be owner-only")
    if path.stat().st_uid != os.geteuid():
        raise ValueError("credential files must be owned by the caller")
    lines = path.read_bytes().splitlines()
    if len(lines) != 1 or not lines[0] or len(lines[0]) > 4096:
        raise ValueError("credential files must contain exactly one bounded line")
    return lines[0].decode("utf-8")


def _request(
    base: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes, str, dict[str, str]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base + path,
        data=body,
        headers=request_headers,
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=60) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
            content_type = response.headers.get("content-type", "")
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_RESPONSE_BYTES + 1)
        status = error.code
        content_type = error.headers.get("content-type", "")
        response_headers = dict(error.headers.items())
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"{path} exceeded the response safety limit")
    return status, raw, content_type, response_headers


def _cloudflare_dns_binding(
    *,
    zone_id: str,
    record_id: str,
    hostname: str,
    target_tunnel: str,
    token: str,
) -> dict[str, Any]:
    path = f"/zones/{zone_id}/dns_records/{record_id}"
    request = urllib.request.Request(
        CF_API_ROOT + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Cloudflare API returned HTTP {error.code}") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Cloudflare API response exceeded the safety limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError("Cloudflare API returned an unsuccessful response")
    record = payload.get("result")
    if not isinstance(record, dict):
        raise RuntimeError("Cloudflare API response lacks a DNS record")
    expected_content = f"{target_tunnel}.cfargotunnel.com"
    if record.get("id") != record_id:
        raise RuntimeError("Cloudflare readback returned a different DNS record ID")
    if record.get("type") != "CNAME" or record.get("name") != hostname:
        raise RuntimeError("public endpoint is not the exact verified CNAME")
    if record.get("proxied") is not True:
        raise RuntimeError("public endpoint CNAME is not proxied")
    if str(record.get("content", "")).rstrip(".").lower() != expected_content:
        raise RuntimeError("public endpoint CNAME does not target the requested tunnel")
    if not isinstance(record.get("modified_on"), str) or not record["modified_on"]:
        raise RuntimeError("public endpoint CNAME lacks a modification timestamp")
    return {
        "zone_id": zone_id,
        "record_id": record_id,
        "hostname": hostname,
        "record_type": "CNAME",
        "proxied": True,
        "content": expected_content,
        "modified_on": record.get("modified_on"),
    }


def _cloudflare_tunnel_ingress_binding(
    *,
    account_id: str,
    target_tunnel: str,
    proof_hostname: str,
    production_hostname: str,
    token: str,
) -> dict[str, Any]:
    path = f"/accounts/{account_id}/cfd_tunnel/{target_tunnel}/configurations"
    request = urllib.request.Request(
        CF_API_ROOT + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Cloudflare API returned HTTP {error.code}") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Cloudflare API response exceeded the safety limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError("Cloudflare API returned an unsuccessful response")
    result = payload.get("result")
    config = result.get("config") if isinstance(result, dict) else None
    ingress = config.get("ingress") if isinstance(config, dict) else None
    if not isinstance(ingress, list) or not ingress:
        raise RuntimeError("Cloudflare tunnel configuration lacks ingress rules")
    expected_service = "http://127.0.0.1:8080"
    bindings: dict[str, dict[str, Any]] = {}
    targets = (proof_hostname, production_hostname)
    for index, rule in enumerate(ingress):
        if not isinstance(rule, dict):
            raise RuntimeError("Cloudflare tunnel ingress rule is malformed")
        hostname = rule.get("hostname")
        for target in targets:
            matches = hostname is None or (
                isinstance(hostname, str) and fnmatch.fnmatchcase(target, hostname)
            )
            if not matches or target in bindings:
                continue
            if (
                hostname != target
                or "path" in rule
                or rule.get("service") != expected_service
            ):
                raise RuntimeError(
                    "Cloudflare tunnel target ingress is shadowed by an earlier rule"
                )
            bindings[target] = {
                "service": expected_service,
                "rule_index": index,
            }
    if set(bindings) != set(targets):
        raise RuntimeError(
            "Cloudflare tunnel lacks the exact proof and production ingress"
        )
    return {
        "account_id": account_id,
        "tunnel_id": target_tunnel,
        "proof_hostname": proof_hostname,
        "production_hostname": production_hostname,
        "service": expected_service,
        "proof_rule_index": bindings[proof_hostname]["rule_index"],
        "production_rule_index": bindings[production_hostname]["rule_index"],
        "config_version": result.get("version"),
    }


def _manifest_identity(path: Path) -> tuple[dict[str, str], str]:
    payload, manifest_sha256 = load_owned_file(path)
    container = payload.get("container")
    if not isinstance(container, dict):
        raise ValueError("release manifest lacks container identity")
    identity = validate(
        payload,
        container_id=str(container.get("id", "")),
        image_digest=str(container.get("image_digest", "")),
        fork_commit=str(payload.get("source_commit", "")),
    )
    return identity, manifest_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--access-client-id-file", type=Path, required=True)
    parser.add_argument("--access-client-secret-file", type=Path, required=True)
    parser.add_argument(
        "--target-mode",
        choices=("qualified-candidate", "recorded-rollback"),
        required=True,
    )
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--target-tunnel", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--stable-record-id", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--production-hostname", required=True)
    parser.add_argument("--expected-current-tunnel", required=True)
    parser.add_argument("--cloudflare-api-token-file", type=Path, required=True)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path:
        parser.error("endpoint must be an HTTPS origin with no path")
    tunnel_pattern = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    )
    if tunnel_pattern.fullmatch(args.target_tunnel) is None:
        parser.error("target tunnel must be an exact lowercase UUID")
    if (
        HEX_ID.fullmatch(args.account_id) is None
        or HEX_ID.fullmatch(args.zone_id) is None
        or HEX_ID.fullmatch(args.record_id) is None
        or HEX_ID.fullmatch(args.stable_record_id) is None
    ):
        parser.error("zone ID and record ID must be exact lowercase 32-character IDs")
    if not re.fullmatch(r"[a-z0-9.-]+", args.hostname) or "." not in args.hostname:
        parser.error("hostname must be an exact lowercase DNS name")
    if (
        not re.fullmatch(r"[a-z0-9.-]+", args.production_hostname)
        or "." not in args.production_hostname
    ):
        parser.error("production hostname must be an exact lowercase DNS name")
    if parsed.hostname != args.hostname or parsed.port is not None:
        parser.error(
            "endpoint must use the exact verified hostname and default HTTPS port"
        )
    if args.record_id == args.stable_record_id:
        parser.error("proof and stable DNS records must be distinct")
    if args.hostname == args.production_hostname:
        parser.error("proof and production hostnames must be distinct")
    if tunnel_pattern.fullmatch(args.expected_current_tunnel) is None:
        parser.error("current tunnel must be an exact lowercase UUID")
    if args.target_tunnel == args.expected_current_tunnel:
        parser.error("current and target tunnels must differ")
    proxy_variables = (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    )
    if any(os.environ.get(name) for name in proxy_variables):
        parser.error("proxy environment overrides are forbidden")

    access = {
        "CF-Access-Client-Id": _secret(args.access_client_id_file),
        "CF-Access-Client-Secret": _secret(args.access_client_secret_file),
    }
    authorized = {**access, "Authorization": f"Bearer {_secret(args.api_key_file)}"}
    cloudflare_token = _secret(args.cloudflare_api_token_file)
    dns_binding = _cloudflare_dns_binding(
        zone_id=args.zone_id,
        record_id=args.record_id,
        hostname=args.hostname,
        target_tunnel=args.target_tunnel,
        token=cloudflare_token,
    )
    route_precondition = _cloudflare_dns_binding(
        zone_id=args.zone_id,
        record_id=args.stable_record_id,
        hostname=args.production_hostname,
        target_tunnel=args.expected_current_tunnel,
        token=cloudflare_token,
    )
    tunnel_ingress_binding = _cloudflare_tunnel_ingress_binding(
        account_id=args.account_id,
        target_tunnel=args.target_tunnel,
        proof_hostname=args.hostname,
        production_hostname=args.production_hostname,
        token=cloudflare_token,
    )
    if args.target_mode == "qualified-candidate":
        if args.release_manifest is None:
            parser.error("qualified-candidate proof requires --release-manifest")
        identity, manifest_sha256 = _manifest_identity(args.release_manifest)
    else:
        if args.release_manifest is not None:
            parser.error("recorded-rollback proof must not claim a release manifest")
        identity, manifest_sha256 = {}, None

    status, _, _, _ = _request(endpoint, "/health")
    if status not in (401, 403):
        raise RuntimeError(f"unauthenticated public health returned HTTP {status}")

    status, _, _, response_headers = _request(endpoint, "/health", headers=access)
    if status != 200:
        raise RuntimeError(f"Access-authenticated health returned HTTP {status}")
    if identity:
        normalized_headers = {
            name.lower(): value for name, value in response_headers.items()
        }
        expected_headers = {
            "x-verse-candidate-container": identity["candidate_container_id"],
            "x-verse-image-digest": identity["image_digest"],
            "x-verse-fork-commit": identity["fork_commit"],
            "x-verse-model-revision": identity["model_revision"],
            "x-verse-release-nonce": identity["release_nonce"],
            "x-verse-release-manifest-sha256": manifest_sha256,
            "x-verse-attestation-verification-sha256": identity[
                "attestation_verification_sha256"
            ],
        }
        for name, expected in expected_headers.items():
            if normalized_headers.get(name) != expected:
                raise RuntimeError(f"public gateway identity header mismatch: {name}")

    status, body, _, _ = _request(endpoint, "/v1/models", headers=authorized)
    if status != 200:
        raise RuntimeError(f"model listing returned HTTP {status}")
    model_ids = {item.get("id") for item in json.loads(body).get("data", [])}
    if model_ids != {args.model}:
        raise RuntimeError("public route exposed an unexpected model identity")

    status, body, _, _ = _request(
        endpoint,
        "/tokenize",
        method="POST",
        headers=authorized,
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with ready."}],
        },
    )
    if status != 200 or not isinstance(json.loads(body).get("count"), int):
        raise RuntimeError("public tokenizer contract failed")

    status, body, content_type, _ = _request(
        endpoint,
        "/v1/chat/completions",
        method="POST",
        headers={**authorized, "Accept": "text/event-stream"},
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": True,
        },
    )
    if status != 200 or "text/event-stream" not in content_type.lower():
        raise RuntimeError("public streaming contract failed")
    if b"data: [DONE]" not in body or b'"content"' not in body:
        raise RuntimeError("public stream is missing content or the DONE terminator")

    for path in FORBIDDEN:
        status, _, _, _ = _request(
            endpoint, path, method="POST", headers=authorized, payload={}
        )
        if status != 404:
            raise RuntimeError(f"forbidden public path {path} returned HTTP {status}")

    print(
        json.dumps(
            {
                "status": "pass",
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "model": args.model,
                "target_mode": args.target_mode,
                "target_tunnel": args.target_tunnel,
                "dns_binding": dns_binding,
                "route_precondition": route_precondition,
                "tunnel_ingress_binding": tunnel_ingress_binding,
                "release_manifest_sha256": manifest_sha256,
                **identity,
                "allowed_paths": 4,
                "forbidden_paths": len(FORBIDDEN),
            },
            sort_keys=True,
        )
    )
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
