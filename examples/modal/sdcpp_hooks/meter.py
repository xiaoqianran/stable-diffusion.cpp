from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from .cost import CostEvent, PriceBook, ResourcePlan, billed_usd, format_event, format_usd, make_event


_TRACE_ID: ContextVar[str] = ContextVar("sdcpp_cost_trace", default="")
_PRICE_BOOK: ContextVar[PriceBook | None] = ContextVar("sdcpp_cost_book", default=None)
_LAST_EVENT: ContextVar[CostEvent | None] = ContextVar("sdcpp_cost_last", default=None)


def current_trace_id() -> str:
    return _TRACE_ID.get()


def current_book() -> PriceBook:
    return _PRICE_BOOK.get() or PriceBook()


def last_event() -> CostEvent | None:
    return _LAST_EVENT.get()


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


@contextmanager
def span(
    plan: ResourcePlan,
    *,
    phase: str,
    book: PriceBook | None = None,
    ledger: Ledger | None = None,
    extra: dict | None = None,
) -> Iterator[None]:
    book = book or current_book()
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        yield
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        event = make_event(
            event_id=uuid.uuid4().hex[:12],
            plan=plan,
            duration_ms=duration_ms,
            book=book,
            phase=phase,
            trace_id=current_trace_id(),
            started_at=started_at,
            extra=extra,
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
