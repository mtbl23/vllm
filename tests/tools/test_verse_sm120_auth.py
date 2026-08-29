# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm"
    / "entrypoints"
    / "serve"
    / "utils"
    / "server_utils.py"
)


def _load_server_utils():
    envs = types.SimpleNamespace(VLLM_VERSE_RUNTIME_STRICT=False)
    vllm = types.ModuleType("vllm")
    vllm.envs = envs
    protocol = types.ModuleType("vllm.engine.protocol")
    protocol.EngineClient = object
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _name: types.SimpleNamespace()
    gc_utils = types.ModuleType("vllm.utils.gc_utils")
    gc_utils.freeze_gc_heap = lambda: None
    stubs = {
        "vllm": vllm,
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.protocol": protocol,
        "vllm.logger": logger,
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.gc_utils": gc_utils,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("verse_server_utils", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module, envs


SERVER_UTILS, ENVS = _load_server_utils()


def _client(strict: bool) -> TestClient:
    ENVS.VLLM_VERSE_RUNTIME_STRICT = strict
    app = FastAPI()
    for path in (
        "/health",
        "/metrics",
        "/tokenize",
        "/invocations",
        "/docs-probe",
        "/v1/models",
    ):
        app.add_api_route(path, lambda: {"status": "ok"}, methods=["GET"])
    app.add_middleware(SERVER_UTILS.AuthenticationMiddleware, tokens=["opaque-key"])
    return TestClient(app)


def test_verse_strict_authenticates_every_application_route():
    client = _client(strict=True)
    authorized = {"Authorization": "Bearer opaque-key"}

    for path in ("/tokenize", "/invocations", "/docs-probe", "/v1/models"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers=authorized).status_code == 200

    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_upstream_authentication_scope_is_unchanged_outside_verse():
    client = _client(strict=False)

    assert client.get("/invocations").status_code == 200
    assert client.get("/tokenize").status_code == 200
    assert client.get("/v1/models").status_code == 401
