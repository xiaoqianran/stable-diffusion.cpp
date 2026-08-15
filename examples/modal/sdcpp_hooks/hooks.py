from __future__ import annotations

from pathlib import Path
from typing import Callable

from .adapt import adapt_request
from .artifacts import FetchFn, TokenFn, is_fetchable, resolve_artifacts
from .contract import GenerateRequest, GenerateResult
from .discover import EngineCapabilities, discover_engine


ProbeFn = Callable[[], str]
RunFn = Callable[[list[str], Path], list[Path]]


def use_engine(
    *,
    probe: ProbeFn | None = None,
    help_text: str | None = None,
    binary: str = "sd-cli",
) -> EngineCapabilities:
    text = help_text if help_text is not None else (probe() if probe else "")
    return discover_engine(text, binary=binary)


def use_models(
    request: GenerateRequest,
    cache_dir: Path,
    fetch: FetchFn | None = None,
    token_for_url: TokenFn | None = None,
) -> dict[str, Path]:
    resolved = resolve_artifacts(
        request.to_dict(),
        cache_dir=cache_dir,
        fetch=fetch,
        token_for_url=token_for_url,
    )
    extra = {
        key: value
        for key, value in request.extra_cli.items()
        if is_fetchable(value)
    }
    if extra:
        resolved.update(
            resolve_artifacts(
                extra,
                cache_dir=cache_dir,
                fetch=fetch,
                token_for_url=token_for_url,
                artifact_fields=set(extra),
            )
        )
    return resolved


def _with_resolved_paths(request: GenerateRequest, models: dict[str, Path]) -> GenerateRequest:
    data = request.to_dict()
    extra = dict(data.get("extra_cli") or {})
    for key, path in models.items():
        if key in extra or str(key).startswith("-"):
            extra[key] = str(path)
        else:
            data[key] = str(path)
    data["extra_cli"] = extra
    return GenerateRequest.from_dict(data)


def generate(
    request: GenerateRequest,
    *,
    engine: EngineCapabilities,
    models: dict[str, Path],
    run: RunFn,
    output_path: Path,
) -> GenerateResult:
    request.validate()
    planned = adapt_request(_with_resolved_paths(request, models), engine, str(output_path))
    files = run(planned.argv, Path(output_path).parent)
    return GenerateResult(
        images=[path.read_bytes() for path in files],
        argv=planned.argv,
        dropped_fields=planned.dropped_fields,
        engine_id=engine.binary,
        seed=request.seed,
    )


def use_sdcpp(
    *,
    probe: ProbeFn,
    binary: str,
    cache_dir: Path,
    run: RunFn,
    output_path: Path,
    fetch: FetchFn | None = None,
    token_for_url: TokenFn | None = None,
) -> Callable[[GenerateRequest], GenerateResult]:
    engine = use_engine(probe=probe, binary=binary)

    def _generate(request: GenerateRequest) -> GenerateResult:
        models = use_models(
            request,
            cache_dir=cache_dir,
            fetch=fetch,
            token_for_url=token_for_url,
        )
        return generate(
            request,
            engine=engine,
            models=models,
            run=run,
            output_path=output_path,
        )

    return _generate
