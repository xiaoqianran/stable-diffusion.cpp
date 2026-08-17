from __future__ import annotations

import os
from typing import Any


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


def _lookup_function(app_name: str, function_name: str):
    import modal

    return modal.Function.from_name(
        app_name,
        function_name,
        environment_name=MODAL_ENVIRONMENT,
    )


def _lookup_cls(app_name: str, cls_name: str):
    import modal

    return modal.Cls.from_name(
        app_name,
        cls_name,
        environment_name=MODAL_ENVIRONMENT,
    )


def _is_deployed(app_name: str, *, cls_name: str | None = None, function_name: str | None = None) -> bool:
    from modal.exception import NotFoundError

    handle = (
        _lookup_cls(app_name, cls_name)
        if cls_name is not None
        else _lookup_function(app_name, function_name or "")
    )
    try:
        handle.hydrate()
    except NotFoundError:
        return False
    return True


def ensure_deployed(*, force: bool | None = None) -> None:
    """Ensure both Modal Apps are persistently deployed, without starting GPU compute."""
    if force is None:
        force = os.environ.get("SDCPP_FORCE_DEPLOY") == "1"

    storage_ready = STORAGE_APP_NAME in _READY and not force
    gpu_ready = GPU_APP_NAME in _READY and not force

    if not storage_ready:
        storage_ready = (not force) and _is_deployed(
            STORAGE_APP_NAME,
            function_name="ensure_artifacts",
        )
    if not gpu_ready:
        gpu_ready = (not force) and _is_deployed(
            GPU_APP_NAME,
            cls_name="SDEngine",
        )

    if storage_ready and gpu_ready:
        _READY.update({STORAGE_APP_NAME, GPU_APP_NAME})
        return

    from app import gpu_app, storage_app

    if not storage_ready:
        storage_app.deploy(
            name=STORAGE_APP_NAME,
            environment_name=MODAL_ENVIRONMENT,
        )
        _READY.add(STORAGE_APP_NAME)
    if not gpu_ready:
        gpu_app.deploy(
            name=GPU_APP_NAME,
            environment_name=MODAL_ENVIRONMENT,
        )
        _READY.add(GPU_APP_NAME)


def storage_function(name: str):
    ensure_deployed()
    return _lookup_function(STORAGE_APP_NAME, name)


def gpu_function(name: str):
    ensure_deployed()
    return _lookup_function(GPU_APP_NAME, name)


def engine(
    *,
    gpu: str | None = None,
    recipe: str = "",
    max_containers: int | None = None,
) -> Any:
    """Return one deployed SDEngine pool.

    Web recipe calls use a stable max_containers=4 variant and control actual
    fan-out client-side (1/2/4). Keeping the variant configuration stable lets
    consecutive jobs with the same recipe/GPU reuse warm model-loaded containers.
    """
    ensure_deployed()
    cls = _lookup_cls(GPU_APP_NAME, "SDEngine")
    options: dict[str, Any] = {
        "max_containers": max_containers or GPU_MAX_CONTAINERS,
    }
    if gpu:
        options["gpu"] = gpu
    cls = cls.with_options(**options)
    return cls(recipe=recipe)
