# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CADDYFILE = ROOT / "tools" / "verse" / "verse-sm120-gateway.Caddyfile"
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


def test_caddy_gateway_has_exact_allowlist_and_no_management_routes():
    text = CADDYFILE.read_text()
    for path in ("/health", "/v1/models", "/tokenize", "/v1/chat/completions"):
        assert f"path {path}" in text
    for path in public.FORBIDDEN:
        assert f"path {path}" not in text
    assert "respond 404" in text
    assert "127.0.0.1:8000" in text
    assert "0.0.0.0" not in text


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
    route._write_receipt(receipt, {"status": "verified"})

    assert json.loads(receipt.read_text()) == {"status": "verified"}
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        route._write_receipt(receipt, {"status": "replaced"})


def test_public_gateway_credential_reader_rejects_broad_permissions(tmp_path: Path):
    credential = tmp_path / "secret"
    credential.write_text("value\n")
    credential.chmod(0o600)
    assert public._secret(credential.resolve()) == "value"

    credential.chmod(0o640)
    with pytest.raises(ValueError, match="owner-only"):
        public._secret(credential.resolve())
