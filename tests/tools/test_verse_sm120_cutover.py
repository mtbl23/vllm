# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CADDYFILE = ROOT / "tools" / "verse" / "verse-sm120-gateway.Caddyfile"
GATEWAY_RUNNER = ROOT / "tools" / "verse" / "run_sm120_gateway.sh"
ATTESTATION_VERIFIER = ROOT / "tools" / "verse" / "verify_sm120_attestation.sh"
DUAL_CADDYFILE = ROOT / "tools" / "verse" / "verse-sm120-dual-gateway.Caddyfile"
DUAL_LAUNCHER = ROOT / "tools" / "verse" / "run_sm120_dual_gateway.sh"
DUAL_VERIFY = ROOT / "tools" / "verse" / "verify_sm120_dual_gateway.py"
ROUTE_PATH = ROOT / "tools" / "verse" / "switch_sm120_cloudflare_route.py"
PUBLIC_PATH = ROOT / "tools" / "verse" / "verify_sm120_public_gateway.py"
VALIDATE_PATH = ROOT / "tools" / "verse" / "validate_sm120_gateway_release.py"
BIND_PATH = ROOT / "tools" / "verse" / "bind_sm120_candidate_release.py"
sys.path.insert(0, str(ROUTE_PATH.parent))

FORK_COMMIT = "b" * 40
IMAGE_DIGEST = f"registry/runtime@sha256:{'a' * 64}"
CONTAINER_ID = "c" * 64
RELEASE_NONCE = "d" * 64
MODEL_REVISION = "e2c6cd9c3302e91c032a378a607009c82ba16fac"
ATTESTATION_SHA256 = "f" * 64
PROOF_ZONE_ID = "1" * 32
PROOF_RECORD_ID = "2" * 32
PROOF_HOSTNAME = "candidate-proof.verse-rp.com"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


route = _load("switch_sm120_cloudflare_route", ROUTE_PATH)
public = _load("verify_sm120_public_gateway", PUBLIC_PATH)
dual = _load("verify_sm120_dual_gateway", DUAL_VERIFY)
release = _load("validate_sm120_gateway_release", VALIDATE_PATH)
binder = _load("bind_sm120_candidate_release", BIND_PATH)


def _release_manifest() -> dict:
    return {
        "status": "pass",
        "scope": "pre_cutover_candidate_qualification",
        "profile": "sm120-gemma4-nvfp4-v4",
        "source_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "release_nonce": RELEASE_NONCE,
        "image_attestation": {
            "image_repository": "registry/runtime",
            "image_sha256": "a" * 64,
            "source_commit": FORK_COMMIT,
            "signer_workflow": ".github/workflows/verse-sm120-image.yml",
            "source_ref": "refs/heads/verse/v0.28-sm120-nvfp4-fa2",
            "verification_sha256": ATTESTATION_SHA256,
        },
        "container": {"id": CONTAINER_ID, "image_digest": IMAGE_DIGEST},
    }


def _write_target_binding(tmp_path: Path, target_tunnel: str) -> tuple[Path, Path]:
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps(_release_manifest(), sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    proof = tmp_path / "public-proof.json"
    proof.write_text(
        json.dumps(
            {
                "status": "pass",
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "target_tunnel": target_tunnel,
                "target_mode": "qualified-candidate",
                "release_manifest_sha256": manifest_sha,
                "candidate_container_id": CONTAINER_ID,
                "image_digest": IMAGE_DIGEST,
                "fork_commit": FORK_COMMIT,
                "model_revision": MODEL_REVISION,
                "release_nonce": RELEASE_NONCE,
                "attestation_verification_sha256": ATTESTATION_SHA256,
                "dns_binding": {
                    "zone_id": PROOF_ZONE_ID,
                    "record_id": PROOF_RECORD_ID,
                    "hostname": PROOF_HOSTNAME,
                    "record_type": "CNAME",
                    "proxied": True,
                    "content": f"{target_tunnel}.cfargotunnel.com",
                    "modified_on": "2026-08-29T00:00:00Z",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return manifest.resolve(), proof.resolve()


def test_caddy_gateway_has_exact_allowlist_and_no_management_routes():
    text = CADDYFILE.read_text()
    for path in ("/health", "/v1/models", "/tokenize", "/v1/chat/completions"):
        assert f"path {path}" in text
    for path in public.FORBIDDEN:
        assert f"path {path}" not in text
    assert "respond 404" in text
    assert "127.0.0.1:8000" in text
    assert "0.0.0.0" not in text
    for header in (
        "X-Verse-Candidate-Container",
        "X-Verse-Image-Digest",
        "X-Verse-Fork-Commit",
        "X-Verse-Model-Revision",
        "X-Verse-Release-Nonce",
        "X-Verse-Release-Manifest-Sha256",
        "X-Verse-Attestation-Verification-Sha256",
    ):
        assert text.count(header) == 4


def test_gateway_is_bound_to_exact_candidate_network_namespace():
    text = GATEWAY_RUNNER.read_text()

    assert '--network "container:$VLLM_CONTAINER_ID"' in text
    assert "--network host" not in text
    assert "docker update --restart" not in text
    assert "--restart no" in text
    assert '{"8000/tcp", "8080/tcp"}' in text
    assert "candidate must publish only 127.0.0.1:8080 for its gateway" in text
    assert "vLLM API key file must be caller-owned with exact mode 0600" in text


def test_attestation_verifier_uses_exact_github_policy():
    text = ATTESTATION_VERIFIER.read_text()

    assert 'gh attestation verify "oci://$IMAGE"' in text
    assert "--repo mtbl23/vllm" in text
    assert (
        "https://github.com/mtbl23/vllm/.github/workflows/verse-sm120-image.yml" in text
    )
    assert "--source-ref refs/heads/verse/v0.28-sm120-nvfp4-fa2" in text
    assert '--source-digest "$COMMIT"' in text
    assert "--deny-self-hosted-runners" in text
    assert "--format json" in text


def test_public_proof_reads_exact_cloudflare_dns_binding(
    monkeypatch: pytest.MonkeyPatch,
):
    tunnel = "11111111-2222-3333-4444-555555555555"
    payload = {
        "success": True,
        "result": {
            "id": PROOF_RECORD_ID,
            "type": "CNAME",
            "name": PROOF_HOSTNAME,
            "content": f"{tunnel}.cfargotunnel.com",
            "proxied": True,
            "modified_on": "2026-08-29T00:00:00Z",
        },
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            del size
            return json.dumps(payload).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url.endswith(
                f"/zones/{PROOF_ZONE_ID}/dns_records/{PROOF_RECORD_ID}"
            )
            assert timeout == 20
            return Response()

    monkeypatch.setattr(public.urllib.request, "build_opener", lambda *args: Opener())
    binding = public._cloudflare_dns_binding(
        zone_id=PROOF_ZONE_ID,
        record_id=PROOF_RECORD_ID,
        hostname=PROOF_HOSTNAME,
        target_tunnel=tunnel,
        token="secret",
    )

    assert binding["content"] == f"{tunnel}.cfargotunnel.com"
    payload["result"]["content"] = "other.cfargotunnel.com"
    with pytest.raises(RuntimeError, match="does not target the requested tunnel"):
        public._cloudflare_dns_binding(
            zone_id=PROOF_ZONE_ID,
            record_id=PROOF_RECORD_ID,
            hostname=PROOF_HOSTNAME,
            target_tunnel=tunnel,
            token="secret",
        )


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
    manifest, proof = _write_target_binding(tmp_path, target_tunnel)
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
            "--target-release-manifest",
            str(manifest),
            "--target-public-proof",
            str(proof),
            "--target-mode",
            "qualified-candidate",
            "--target-proof-zone-id",
            PROOF_ZONE_ID,
            "--target-proof-record-id",
            PROOF_RECORD_ID,
            "--target-proof-hostname",
            PROOF_HOSTNAME,
            "--apply",
        ],
    )

    assert route.main() == 0
    assert calls == ["GET", "GET", "PATCH", "GET"]
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "verified"
    assert saved["target_binding"]["candidate_container_id"] == CONTAINER_ID
    assert saved["target_binding"]["target_tunnel"] == target_tunnel


def test_target_binding_rejects_proof_for_another_manifest(tmp_path: Path):
    tunnel = "66666666-7777-8888-9999-aaaaaaaaaaaa"
    manifest, proof = _write_target_binding(tmp_path, tunnel)
    payload = json.loads(proof.read_text())
    payload["release_manifest_sha256"] = "f" * 64
    proof.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not cover the target release manifest"):
        route._validate_target_binding(
            manifest_path=manifest,
            proof_path=proof,
            target_tunnel=tunnel,
            target_mode="qualified-candidate",
            proof_zone_id=PROOF_ZONE_ID,
            proof_record_id=PROOF_RECORD_ID,
            proof_hostname=PROOF_HOSTNAME,
        )


def test_recorded_rollback_requires_fresh_public_service_proof(tmp_path: Path):
    tunnel = "66666666-7777-8888-9999-aaaaaaaaaaaa"
    proof = tmp_path / "rollback-proof.json"
    proof.write_text(
        json.dumps(
            {
                "status": "pass",
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "target_mode": "recorded-rollback",
                "target_tunnel": tunnel,
                "endpoint": "https://old-candidate.example.com",
                "model": "verse-free",
                "release_manifest_sha256": None,
                "dns_binding": {
                    "zone_id": PROOF_ZONE_ID,
                    "record_id": PROOF_RECORD_ID,
                    "hostname": PROOF_HOSTNAME,
                    "record_type": "CNAME",
                    "proxied": True,
                    "content": f"{tunnel}.cfargotunnel.com",
                },
            }
        )
    )
    binding = route._validate_target_binding(
        target_mode="recorded-rollback",
        manifest_path=None,
        proof_path=proof.resolve(),
        target_tunnel=tunnel,
        proof_zone_id=PROOF_ZONE_ID,
        proof_record_id=PROOF_RECORD_ID,
        proof_hostname=PROOF_HOSTNAME,
    )
    assert binding["target_mode"] == "recorded-rollback"
    assert binding["model"] == "verse-free"


def test_deployment_state_rejects_group_writable_parent(tmp_path: Path):
    parent = tmp_path / "unsafe"
    parent.mkdir(mode=0o770)
    parent.chmod(0o770)

    with pytest.raises(ValueError, match="group/world writable"):
        route._reserve_receipt(parent / "receipt.json", {"status": "pending"})


def test_deployment_state_rejects_group_writable_ancestor(tmp_path: Path):
    ancestor = tmp_path / "unsafe-ancestor"
    parent = ancestor / "safe-parent"
    parent.mkdir(parents=True, mode=0o700)
    ancestor.chmod(0o770)

    with pytest.raises(ValueError, match="ancestry is group/world writable"):
        route._reserve_receipt(parent / "receipt.json", {"status": "pending"})


def test_release_manifest_binds_exact_candidate(tmp_path: Path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps(_release_manifest()))
    payload, digest = release.load_owned_file(manifest.resolve())
    identity = release.validate(
        payload,
        container_id=CONTAINER_ID,
        image_digest=IMAGE_DIGEST,
        fork_commit=FORK_COMMIT,
    )

    assert len(digest) == 64
    assert identity["candidate_container_id"] == CONTAINER_ID
    assert identity["attestation_verification_sha256"] == ATTESTATION_SHA256
    with pytest.raises(ValueError, match="wrong container"):
        release.validate(
            payload,
            container_id="e" * 64,
            image_digest=IMAGE_DIGEST,
            fork_commit=FORK_COMMIT,
        )


def test_candidate_binding_requires_passed_image_qualification():
    qualification = {
        "status": "pass",
        "scope": "disposable_image_qualification",
        "profile": "sm120-gemma4-nvfp4-v4",
        "image_digest": IMAGE_DIGEST,
        "source_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "image_attestation": _release_manifest()["image_attestation"],
        "b01": {"status": "pass"},
    }
    candidate = {
        "status": "valid",
        "container_id": CONTAINER_ID,
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "gateway_host_port": 8080,
        "restart_policy": "no",
    }

    result = binder.bind(
        qualification,
        "1" * 64,
        candidate,
        "2" * 64,
        container_id=CONTAINER_ID,
        image=IMAGE_DIGEST,
        commit=FORK_COMMIT,
    )

    assert result["scope"] == "pre_cutover_candidate_binding"
    assert result["qualification_manifest_sha256"] == "1" * 64
    assert result["candidate_validation_sha256"] == "2" * 64
    with pytest.raises(ValueError, match="candidate validation drifted"):
        binder.bind(
            qualification,
            "1" * 64,
            {**candidate, "restart_policy": "unless-stopped"},
            "2" * 64,
            container_id=CONTAINER_ID,
            image=IMAGE_DIGEST,
            commit=FORK_COMMIT,
        )


def test_public_gateway_credential_reader_rejects_broad_permissions(tmp_path: Path):
    credential = tmp_path / "secret"
    credential.write_text("value\n")
    credential.chmod(0o600)
    assert public._secret(credential.resolve()) == "value"

    credential.chmod(0o640)
    with pytest.raises(ValueError, match="owner-only"):
        public._secret(credential.resolve())
