# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CADDYFILE = ROOT / "tools" / "verse" / "verse-sm120-gateway.Caddyfile"
DUAL_CADDYFILE = ROOT / "tools" / "verse" / "verse-sm120-dual-gateway.Caddyfile"
DUAL_LAUNCHER = ROOT / "tools" / "verse" / "run_sm120_dual_gateway.sh"
DUAL_VERIFY = ROOT / "tools" / "verse" / "verify_sm120_dual_gateway.py"
ROUTE_PATH = ROOT / "tools" / "verse" / "switch_sm120_cloudflare_route.py"
PUBLIC_PATH = ROOT / "tools" / "verse" / "verify_sm120_public_gateway.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


route = _load("switch_sm120_cloudflare_route", ROUTE_PATH)
public = _load("verify_sm120_public_gateway", PUBLIC_PATH)
dual = _load("verify_sm120_dual_gateway", DUAL_VERIFY)


def test_caddy_gateway_has_exact_allowlist_and_no_management_routes():
    text = CADDYFILE.read_text()
    for path in ("/health", "/v1/models", "/tokenize", "/v1/chat/completions"):
        assert f"path {path}" in text
    for path in public.FORBIDDEN:
        assert f"path {path}" not in text
    assert "respond 404" in text
    assert "127.0.0.1:8000" in text
    assert "0.0.0.0" not in text


def test_dual_gateway_is_loopback_sticky_and_fail_closed():
    text = DUAL_CADDYFILE.read_text()

    assert "bind 127.0.0.1" in text
    assert "{$VERSE_GATEWAY_BIND:127.0.0.1:8080}" in text
    assert "{$VERSE_GPU0_ORIGIN:127.0.0.1:18001}" in text
    assert "{$VERSE_GPU1_ORIGIN:127.0.0.1:18002}" in text
    assert text.count("header X-Verse-Worker gpu0") == 2
    assert text.count("header X-Verse-Worker gpu1") == 2
    assert text.count("header X-Verse-Gateway-Token {$VERSE_GATEWAY_TOKEN}") == 4
    assert text.count('header_up Authorization "Bearer {$VERSE_GPU0_API_KEY}"') == 2
    assert text.count('header_up Authorization "Bearer {$VERSE_GPU1_API_KEY}"') == 2
    for sensitive_header in (
        "CF-Access-Client-Id",
        "CF-Access-Client-Secret",
        "Cf-Access-Jwt-Assertion",
        "Cookie",
        "Proxy-Authorization",
        "X-Verse-Worker",
        "X-Verse-Gateway-Token",
    ):
        assert text.count(f"header_up -{sensitive_header}") == 4
    for path in dual.FORBIDDEN_PATHS:
        if path != "/":
            assert f"path {path}" not in text
    assert "respond 404" in text
    assert "0.0.0.0" not in text


def test_dual_gateway_launcher_pins_binary_and_drops_privileges():
    text = DUAL_LAUNCHER.read_text()

    assert "VERSE_CADDY_SHA256" in text
    assert 'sha256sum "$VERSE_CADDY_BINARY"' in text
    assert "gateway bind port is already listening" in text
    assert "dual workers must use distinct API keys" in text
    assert "VERSE_GATEWAY_TOKEN_FILE" in text
    assert "gateway credential must be distinct from worker API keys" in text
    assert "gateway key file owner does not match its pinned UID" in text
    assert "VERSE_GPU0_KEY_UID" in text
    assert "VERSE_GPU1_KEY_UID" in text
    assert "gateway UID must be distinct from worker key owners" in text
    assert "ulimit -c 0" in text
    assert "/usr/bin/setpriv" in text
    assert "--no-new-privs" in text
    assert "--clear-groups" in text


def test_cloudflare_route_accepts_only_exact_proxied_record():
    tunnel_id = "11111111-2222-3333-4444-555555555555"
    content = route._tunnel_cname(tunnel_id)
    record = {
        "id": "a" * 32,
        "type": "CNAME",
        "name": "free-inference.verse-rp.com",
        "content": content + ".",
        "proxied": True,
    }
    route._validate_record(
        record,
        record_id="a" * 32,
        hostname="free-inference.verse-rp.com",
        expected_content=content,
    )

    for field, value in (
        ("id", "b" * 32),
        ("type", "A"),
        ("name", "other.example.com"),
        ("content", "elsewhere.example.com"),
        ("proxied", False),
    ):
        malformed = dict(record)
        malformed[field] = value
        with pytest.raises(RuntimeError):
            route._validate_record(
                malformed,
                record_id="a" * 32,
                hostname="free-inference.verse-rp.com",
                expected_content=content,
            )


def test_cloudflare_route_receipt_is_exclusive_and_owner_only(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    fd = route._reserve_receipt(receipt, {"status": "pending"})
    route._finish_receipt(fd, {"status": "verified"})
    os.close(fd)

    assert json.loads(receipt.read_text()) == {"status": "verified"}
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        route._reserve_receipt(receipt, {"status": "replaced"})


def test_cloudflare_route_reserves_receipt_and_rechecks_before_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    token = tmp_path / "token"
    token.write_text("secret\n")
    token.chmod(0o600)
    receipt = tmp_path / "receipt.json"
    lock = tmp_path / "cutover.lock"
    record_id = "a" * 32
    current_tunnel = "11111111-2222-3333-4444-555555555555"
    target_tunnel = "66666666-7777-8888-9999-aaaaaaaaaaaa"
    current = route._tunnel_cname(current_tunnel)
    target = route._tunnel_cname(target_tunnel)
    calls: list[str] = []

    def request(*, method, path, token, payload=None):
        del path, token
        calls.append(method)
        if method == "PATCH":
            assert receipt.exists()
            assert json.loads(receipt.read_text())["status"] == "pending"
            assert payload == {"content": target}
            content = target
            modified = "2026-08-29T00:00:01Z"
        else:
            content = target if calls.count("GET") == 3 else current
            modified = (
                "2026-08-29T00:00:01Z"
                if calls.count("GET") == 3
                else "2026-08-29T00:00:00Z"
            )
        return {
            "id": record_id,
            "type": "CNAME",
            "name": "free-inference.verse-rp.com",
            "content": content,
            "proxied": True,
            "modified_on": modified,
        }

    monkeypatch.setattr(route, "_api_request", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROUTE_PATH),
            "switch",
            "--zone-id",
            "b" * 32,
            "--record-id",
            record_id,
            "--hostname",
            "free-inference.verse-rp.com",
            "--expected-current-tunnel",
            current_tunnel,
            "--target-tunnel",
            target_tunnel,
            "--api-token-file",
            str(token.resolve()),
            "--receipt",
            str(receipt.resolve()),
            "--lock",
            str(lock.resolve()),
            "--apply",
        ],
    )

    assert route.main() == 0
    assert calls == ["GET", "GET", "PATCH", "GET"]
    assert json.loads(receipt.read_text())["status"] == "verified"


def test_public_gateway_credential_reader_rejects_broad_permissions(tmp_path: Path):
    credential = tmp_path / "secret"
    credential.write_text("value\n")
    credential.chmod(0o600)
    assert public._secret(credential.resolve()) == "value"

    credential.chmod(0o640)
    with pytest.raises(ValueError, match="owner-only"):
        public._secret(credential.resolve())
