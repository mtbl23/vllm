#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Fail-closed public-route smoke test for the Verse SM120 gateway."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 1_000_000
FORBIDDEN = (
    "/pause",
    "/abort_requests",
    "/invocations",
    "/metrics",
    "/docs",
    "/openapi.json",
)


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
) -> tuple[int, bytes, str]:
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
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=60) as response:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--access-client-id-file", type=Path, required=True)
    parser.add_argument("--access-client-secret-file", type=Path, required=True)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path:
        parser.error("endpoint must be an HTTPS origin with no path")
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

    status, _, _ = _request(endpoint, "/health")
    if status not in (401, 403):
        raise RuntimeError(f"unauthenticated public health returned HTTP {status}")

    status, _, _ = _request(endpoint, "/health", headers=access)
    if status != 200:
        raise RuntimeError(f"Access-authenticated health returned HTTP {status}")

    status, body, _ = _request(endpoint, "/v1/models", headers=authorized)
    if status != 200:
        raise RuntimeError(f"model listing returned HTTP {status}")
    model_ids = {item.get("id") for item in json.loads(body).get("data", [])}
    if model_ids != {args.model}:
        raise RuntimeError("public route exposed an unexpected model identity")

    status, body, _ = _request(
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

    status, body, content_type = _request(
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
        status, _, _ = _request(
            endpoint, path, method="POST", headers=authorized, payload={}
        )
        if status != 404:
            raise RuntimeError(f"forbidden public path {path} returned HTTP {status}")

    print(
        json.dumps(
            {
                "status": "pass",
                "endpoint": endpoint,
                "model": args.model,
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
