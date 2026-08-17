from __future__ import annotations

import os
from typing import Any

from .runtime_identity import local_git_sha


STORAGE_APP_NAME = os.environ.get("SDCPP_STORAGE_APP", "sdcpp-storage")
GPU_APP_NAME = os.environ.get("SDCPP_GPU_APP", "sdcpp-cli")
MODAL_ENVIRONMENT = os.environ.get("SDCPP_MODAL_ENV") or None


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


GPU_MAX_CONTAINERS = _positive_int_env("SDCPP_GPU_MAX_CONTAINERS", 1)
WEB_GPU_POOL_MAX = _positive_int_env("SDCPP_WEB_GPU_POOL_MAX", 4)
_READY: set[str] = set()
_STATUS: dict[str, Any] = {}


def _lookup_function(app_name: str, function_name: str):
    import modal
    return modal.Function.from_name(app_name, function_name, environment_name=MODAL_ENVIRONMENT)


def _lookup_cls(app_name: str, cls_name: str):
    import modal
    return modal.Cls.from_name(app_name, cls_name, environment_name=MODAL_ENVIRONMENT)


def _is_deployed(app_name: str, *, cls_name: str | None = None, function_name: str | None = None) -> bool:
    from modal.exception import NotFoundError
    handle = _lookup_cls(app_name, cls_name) if cls_name is not None else _lookup_function(app_name, function_name or "")
    try:
        handle.hydrate()
    except NotFoundError:
        return False
    return True


def _remote_identity(app_name: str, function_name: str) -> dict[str, Any] | None:
    # Do not turn a transient control-plane/network failure into an automatic
    # redeploy. Missing identity endpoints are detected separately with hydrate().
    value = _lookup_function(app_name, function_name).remote()
    return value if isinstance(value, dict) else None


def expected_deploy_sha() -> str:
    return os.environ.get("SDCPP_DEPLOY_SHA") or local_git_sha()


def deployment_status(*, refresh: bool = False) -> dict[str, Any]:
    if _STATUS and not refresh:
        return dict(_STATUS)
    expected = expected_deploy_sha()
    storage = _remote_identity(STORAGE_APP_NAME, "deployment_info")
    gpu = _remote_identity(GPU_APP_NAME, "gpu_deployment_info")
    status = {
        "expected_sha": expected,
        "storage": storage,
        "gpu": gpu,
        "matches": bool(
            storage and gpu and (not expected or (storage.get("deploy_sha") == expected and gpu.get("deploy_sha") == expected))
        ),
    }
    _STATUS.clear(); _STATUS.update(status)
    return dict(status)


def ensure_deployed(*, force: bool | None = None) -> None:
    """Ensure both Apps exist and match the local source revision when known."""
    if force is None:
        force = os.environ.get("SDCPP_FORCE_DEPLOY") == "1"

    # Once this process has verified/deployed both apps for its expected SHA,
    # avoid two identity RPCs on every subsequent request. Doctor can still call
    # deployment_status(refresh=True) explicitly when a fresh control-plane check
    # is desired.
    if not force and STORAGE_APP_NAME in _READY and GPU_APP_NAME in _READY:
        return

    storage_ready = False
    gpu_ready = False
    if not storage_ready:
        storage_ready = (not force) and _is_deployed(STORAGE_APP_NAME, function_name="ensure_artifacts")
    if not gpu_ready:
        gpu_ready = (not force) and _is_deployed(GPU_APP_NAME, cls_name="SDEngine")

    expected = expected_deploy_sha()
    if storage_ready and gpu_ready and expected:
        # Older deployments do not have identity endpoints and are rolled forward.
        # A transient failure while reading an existing identity endpoint is
        # allowed to propagate rather than causing an accidental deployment.
        identities_exist = (
            _is_deployed(STORAGE_APP_NAME, function_name="deployment_info")
            and _is_deployed(GPU_APP_NAME, function_name="gpu_deployment_info")
        )
        if not identities_exist:
            storage_ready = gpu_ready = False
        else:
            status = deployment_status(refresh=True)
            if not status["matches"]:
                storage_ready = gpu_ready = False

    if storage_ready and gpu_ready:
        _READY.update({STORAGE_APP_NAME, GPU_APP_NAME})
        return

    if expected:
        os.environ["SDCPP_DEPLOY_SHA"] = expected
    from app import gpu_app, storage_app
    if not storage_ready:
        storage_app.deploy(name=STORAGE_APP_NAME, environment_name=MODAL_ENVIRONMENT)
        _READY.add(STORAGE_APP_NAME)
    if not gpu_ready:
        gpu_app.deploy(name=GPU_APP_NAME, environment_name=MODAL_ENVIRONMENT)
        _READY.add(GPU_APP_NAME)
    _STATUS.clear()


def storage_function(name: str):
    ensure_deployed(); return _lookup_function(STORAGE_APP_NAME, name)


def gpu_function(name: str):
    ensure_deployed(); return _lookup_function(GPU_APP_NAME, name)


def engine(*, gpu: str | None = None, recipe: str = "", max_containers: int | None = None) -> Any:
    """Return a stable recipe/GPU pool; client-side fan-out controls actual 1/2/4 use."""
    ensure_deployed()
    cls = _lookup_cls(GPU_APP_NAME, "SDEngine")
    options: dict[str, Any] = {"max_containers": max_containers or GPU_MAX_CONTAINERS}
    if gpu:
        options["gpu"] = gpu
    cls = cls.with_options(**options)
    return cls(recipe=recipe, gpu_name=gpu or os.environ.get("SDCPP_GPU", "L40S"))
