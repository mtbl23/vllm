#!/usr/bin/env python3
"""Fail-closed proof for the loopback-only Verse dual-worker gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 1_000_000
WORKERS = ("gpu0", "gpu1")
FORBIDDEN_PATHS = (
    "/",
    "/pause",
    "/abort_requests",
    "/invocations",
    "/metrics",
    "/docs",
    "/openapi.json",
)
ALLOWED_REQUESTS = (
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("POST", "/tokenize"),
    ("POST", "/v1/chat/completions"),
)


def request(
    base: str,
    path: str,
    *,
    method: str = "GET",
    worker: str | None = None,
    gateway_token: str | None = None,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes, str]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if worker is not None:
        request_headers["X-Verse-Worker"] = worker
    if gateway_token is not None:
        request_headers["X-Verse-Gateway-Token"] = gateway_token
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    item = urllib.request.Request(
        base + path,
        data=body,
        headers=request_headers,
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(item, timeout=90) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
            content_type = response.headers.get("content-type", "")
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_RESPONSE_BYTES + 1)
        status = error.code
        content_type = error.headers.get("content-type", "")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"{path} exceeded the response safety limit")
    return status, raw, content_type


def verify_worker(base: str, model: str, worker: str, gateway_token: str) -> None:
    status, _, _ = request(
        base, "/health", worker=worker, gateway_token=gateway_token
    )
    if status != 200:
        raise RuntimeError(f"{worker} health returned HTTP {status}")

    status, body, _ = request(
        base,
        "/v1/models",
        worker=worker,
        gateway_token=gateway_token,
        headers={"Authorization": "Bearer attacker-controlled-value"},
    )
    if status != 200:
        raise RuntimeError(f"{worker} model listing returned HTTP {status}")
    model_ids = {item.get("id") for item in json.loads(body).get("data", [])}
    if model_ids != {model}:
        raise RuntimeError(f"{worker} exposed an unexpected model identity")

    status, body, _ = request(
        base,
        "/tokenize",
        method="POST",
        worker=worker,
        gateway_token=gateway_token,
        payload={
            "model": model,
            "messages": [
                {"role": "user", "content": "Synthetic gateway verification."}
            ],
        },
    )
    if status != 200 or not isinstance(json.loads(body).get("count"), int):
        raise RuntimeError(f"{worker} tokenizer contract failed")

    status, body, content_type = request(
        base,
        "/v1/chat/completions",
        method="POST",
        worker=worker,
        gateway_token=gateway_token,
        headers={"Accept": "text/event-stream"},
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": True,
        },
    )
    if status != 200 or "text/event-stream" not in content_type.lower():
        raise RuntimeError(f"{worker} streaming contract failed")
    if b"data: [DONE]" not in body or b'"content"' not in body:
        raise RuntimeError(f"{worker} stream lacks content or DONE")


def verify_fail_closed(base: str, gateway_token: str) -> None:
    for method, path in ALLOWED_REQUESTS:
        payload = {} if method == "POST" else None
        for worker in (None, "gpu9", "GPU0", "gpu0,gpu1"):
            status, _, _ = request(
                base,
                path,
                method=method,
                worker=worker,
                gateway_token=gateway_token,
                payload=payload,
            )
            if status != 404:
                label = "missing" if worker is None else worker
                raise RuntimeError(
                    f"{method} {path} with worker {label} returned HTTP {status}"
                )
    for path in FORBIDDEN_PATHS:
        status, _, _ = request(
            base,
            path,
            method="POST",
            worker="gpu0",
            gateway_token=gateway_token,
            payload={},
        )
        if status != 404:
            raise RuntimeError(f"forbidden path {path} returned HTTP {status}")
    for method, path in (
        ("POST", "/health"),
        ("POST", "/v1/models"),
        ("GET", "/tokenize"),
        ("GET", "/v1/chat/completions"),
    ):
        status, _, _ = request(
            base,
            path,
            method=method,
            worker="gpu0",
            gateway_token=gateway_token,
            payload=None,
        )
        if status != 404:
            raise RuntimeError(f"wrong method {method} {path} returned HTTP {status}")
    for supplied in (None, "wrong-gateway-token"):
        status, _, _ = request(
            base,
            "/health",
            worker="gpu0",
            gateway_token=supplied,
        )
        if status != 404:
            label = "missing" if supplied is None else "invalid"
            raise RuntimeError(
                f"health with {label} gateway credential returned HTTP {status}"
            )


def secret(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("gateway credential path must be an absolute regular file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("gateway credential path must be canonical")
    metadata = resolved.stat()
    if metadata.st_mode & 0o077:
        raise ValueError("gateway credential file must be owner-only")
    raw = resolved.read_bytes()
    if len(raw) > 4096 or b"\x00" in raw or b"\r" in raw:
        raise ValueError("gateway credential contains invalid bytes")
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("gateway credential must contain one non-empty line")
    value = lines[0].decode("utf-8")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError("gateway credential must be printable ASCII without spaces")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--gateway-token-file", type=Path, required=True)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        parser.error("endpoint must be an explicit loopback HTTP origin with a port")
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

    gateway_token = secret(args.gateway_token_file)
    for worker in WORKERS:
        verify_worker(endpoint, args.model, worker, gateway_token)
    verify_fail_closed(endpoint, gateway_token)
    print(
        json.dumps(
            {
                "status": "pass",
                "endpoint": endpoint,
                "model": args.model,
                "workers": list(WORKERS),
                "allowed_routes_per_worker": len(ALLOWED_REQUESTS),
                "forbidden_paths": len(FORBIDDEN_PATHS),
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
