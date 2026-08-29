#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Prove the raw loopback server's strict authentication boundary."""

from __future__ import annotations

import argparse
import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path


def status(endpoint: str, path: str, *, headers: dict[str, str] | None = None) -> int:
    request = urllib.request.Request(endpoint + path, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            response.read(1024)
            return response.status
    except urllib.error.HTTPError as error:
        error.read(1024)
        return error.code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if not endpoint.startswith("http://127.0.0.1:"):
        parser.error("endpoint must be a loopback HTTP origin")
    path = args.api_key_file
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or path.stat().st_uid != os.geteuid()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        parser.error("API key must be a canonical owner-only file")
    lines = path.read_bytes().splitlines()
    if len(lines) != 1 or not lines[0]:
        parser.error("API key must contain one non-empty line")
    authorized = {"Authorization": f"Bearer {lines[0].decode('utf-8')}"}
    unauthenticated = {
        route: status(endpoint, route)
        for route in (
            "/invocations",
            "/tokenize",
            "/docs",
            "/openapi.json",
            "/v1/models",
        )
    }
    liveness = {route: status(endpoint, route) for route in ("/health", "/metrics")}
    if set(unauthenticated.values()) != {401}:
        raise SystemExit("an unauthenticated application route bypassed the API key")
    if set(liveness.values()) != {200}:
        raise SystemExit("an unguarded liveness route is unhealthy")
    if status(endpoint, "/v1/models", headers=authorized) != 200:
        raise SystemExit("authorized model discovery failed")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "unauthenticated": unauthenticated,
                "liveness": liveness,
                "authorized_model_discovery": 200,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
