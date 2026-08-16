from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from .cost import CostEvent, PriceBook, ResourcePlan, billed_usd, format_event, format_usd, make_event


_TRACE_ID: ContextVar[str] = ContextVar("sdcpp_cost_trace", default="")
_PRICE_BOOK: ContextVar[PriceBook | None] = ContextVar("sdcpp_cost_book", default=None)
_LAST_EVENT: ContextVar[CostEvent | None] = ContextVar("sdcpp_cost_last", default=None)
_PARENT_ID: ContextVar[str] = ContextVar("sdcpp_cost_parent", default="")
_TASK: ContextVar[dict] = ContextVar("sdcpp_cost_task", default={})


def current_trace_id() -> str:
    return _TRACE_ID.get()


def current_book() -> PriceBook:
    return _PRICE_BOOK.get() or PriceBook()


def last_event() -> CostEvent | None:
    return _LAST_EVENT.get()


def current_task() -> dict:
    return dict(_TASK.get() or {})


def current_parent_id() -> str:
    return _PARENT_ID.get()


@contextmanager
def bind_task(**fields: Any) -> Iterator[dict]:
    merged = {**current_task(), **{key: value for key, value in fields.items() if value not in (None, "")}}
    token = _TASK.set(merged)
    try:
        yield merged
    finally:
        _TASK.reset(token)


@contextmanager
def bind_parent(parent_id: str) -> Iterator[str]:
    token = _PARENT_ID.set(parent_id)
    try:
        yield parent_id
    finally:
        _PARENT_ID.reset(token)


def client_ledger_path() -> Path:
    raw = os.environ.get("SDCPP_COST_LOG")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "sdcpp-modal" / "cost.jsonl"


def worker_ledger_path() -> Path | None:
    raw = os.environ.get("SDCPP_COST_WORKER_LOG")
    if raw:
        return Path(raw)
    root = Path(os.environ.get("SDCPP_MODEL_ROOT", "/models"))
    if not root.exists() or not os.access(root, os.W_OK):
        return None
    return root / ".sdcpp-cost" / "events.jsonl"


class Ledger:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: CostEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read(self) -> list[CostEvent]:
        if not self.path.is_file():
            return []
        events: list[CostEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(CostEvent.from_dict(json.loads(line)))
        return events


def client_ledger() -> Ledger:
    return Ledger(client_ledger_path())


def worker_ledger() -> Ledger | None:
    path = worker_ledger_path()
    return Ledger(path) if path else None


def _task_fields(extra: dict | None = None) -> dict[str, Any]:
    task = current_task()
    extra = dict(extra or {})
    merged = {**task, **extra}
    if "parent_id" in extra:
        parent_id = str(extra.get("parent_id") or "")
    else:
        parent_id = str(merged.get("parent_id") or current_parent_id() or "")
    return {
        "extra": merged,
        "job_id": str(merged.get("job_id") or ""),
        "image_id": str(merged.get("image_id") or ""),
        "recipe": str(merged.get("recipe") or ""),
        "parent_id": parent_id,
    }


def record_event(
    plan: ResourcePlan,
    *,
    phase: str,
    duration_ms: int = 0,
    book: PriceBook | None = None,
    ledger: Ledger | None = None,
    extra: dict | None = None,
    event_id: str | None = None,
) -> CostEvent:
    event = make_event(
        event_id=event_id or uuid.uuid4().hex[:12],
        plan=plan,
        duration_ms=duration_ms,
        book=book or current_book(),
        phase=phase,
        trace_id=current_trace_id(),
        **_task_fields(extra),
    )
    _LAST_EVENT.set(event)
    dest = ledger if ledger is not None else client_ledger()
    try:
        dest.append(event)
    except OSError:
        pass
    return event


@contextmanager
def span(
    plan: ResourcePlan,
    *,
    phase: str,
    book: PriceBook | None = None,
    ledger: Ledger | None = None,
    extra: dict | None = None,
    event_id: str | None = None,
) -> Iterator[str]:
    book = book or current_book()
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    event_id = event_id or uuid.uuid4().hex[:12]
    try:
        yield event_id
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        event = make_event(
            event_id=event_id,
            plan=plan,
            duration_ms=duration_ms,
            book=book,
            phase=phase,
            trace_id=current_trace_id(),
            started_at=started_at,
            **_task_fields(extra),
        )
        _LAST_EVENT.set(event)
        if ledger is not None:
            try:
                ledger.append(event)
            except OSError:
                pass


class ContainerMeter:
    """GPU/CPU container lifetime. Start in @modal.enter, stop in @modal.exit."""

    def __init__(self, plan: ResourcePlan, ledger: Ledger | None = None):
        self._plan = plan
        self._ledger = ledger if ledger is not None else worker_ledger()
        self._started = time.perf_counter()
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.event: CostEvent | None = None

    @classmethod
    def start(cls, name: str, *, gpu: str | None = None) -> ContainerMeter:
        return cls(ResourcePlan(name=name, gpu=gpu))

    def stop(self) -> CostEvent:
        duration_ms = int((time.perf_counter() - self._started) * 1000)
        self.event = make_event(
            event_id=uuid.uuid4().hex[:12],
            plan=self._plan,
            duration_ms=duration_ms,
            book=current_book(),
            phase="container",
            trace_id=current_trace_id(),
            started_at=self._started_at,
            **_task_fields(),
        )
        if self._ledger is not None:
            try:
                self._ledger.append(self.event)
            except OSError:
                pass
        _LAST_EVENT.set(self.event)
        return self.event


def begin_trace(trace_id: str, book: PriceBook | None = None) -> tuple[object, object]:
    return _TRACE_ID.set(trace_id), _PRICE_BOOK.set(book or PriceBook())


def end_trace(trace_token, book_token) -> None:
    _TRACE_ID.reset(trace_token)
    _PRICE_BOOK.reset(book_token)


def summarize_events(events: list[CostEvent]) -> str:
    if not events:
        return "cost ledger is empty"
    lines = [format_event(event) for event in events[-20:]]
    lines.append(f"cost billed {format_usd(billed_usd(events[-20:]))}  (last {min(20, len(events))} events)")
    return "\n".join(lines)
