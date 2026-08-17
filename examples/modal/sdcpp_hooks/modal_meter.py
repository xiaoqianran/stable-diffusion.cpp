"""Modal-facing cost hooks.

Keep this file as the only place that talks to the Modal SDK for billing.
Deployed service calls are metered here without opening ephemeral app.run()
sessions. `sdcpp_modal.py` and the Web generator look up deployed handles via
`sdcpp_hooks.deployed` and call `.remote()` directly.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .cost import (
    PriceBook,
    ResourcePlan,
    billed_usd,
    default_plan,
    format_event,
    format_per_second,
    format_usd,
)
from .cost_view import format_trace_tree, ledger_report
from .meter import (
    begin_trace,
    bind_parent,
    client_ledger,
    current_book,
    end_trace,
    last_event,
    span,
    summarize_events,
)


def _gpu_name() -> str:
    return os.environ.get("SDCPP_GPU", "L40S")


def plan_for(name: str, *, gpu: bool = False, notes: str = "") -> ResourcePlan:
    return default_plan(name, gpu=_gpu_name() if gpu else None, notes=notes)


def official_price_book() -> PriceBook:
    try:
        from modal import Workspace

        rates = Workspace.from_context().billing.rates()
        return PriceBook({key: str(value) for key, value in rates.items()}, source="modal-billing-rates")
    except Exception:
        return PriceBook(source="fallback")


@contextmanager
def billed_service(role: str) -> Iterator[str]:
    """Meter a window that calls an already-deployed Modal service.

    Unlike the old billed_app() implementation this never enters app.run(), so
    it does not create an ephemeral App for every local request.
    """
    trace_id = uuid.uuid4().hex[:12]
    session_id = uuid.uuid4().hex[:12]
    book = official_price_book()
    tokens = begin_trace(trace_id, book)
    plan = default_plan(
        role,
        notes="deployed service call window; remote/container compute is priced separately",
    )
    try:
        with span(
            plan,
            phase="session",
            book=book,
            ledger=client_ledger(),
            event_id=session_id,
            extra={"role": role, "call": f"deployed:{role}", "parent_id": ""},
        ):
            with bind_parent(session_id):
                yield trace_id
    finally:
        end_trace(*tokens)


@contextmanager
def billed_app(app: Any, role: str) -> Iterator[str]:
    """Backward-compatible alias for callers migrating off ephemeral app.run()."""
    del app
    with billed_service(role) as trace_id:
        yield trace_id


def billed_remote(fn: Any, *args: Any, name: str, gpu: bool = False, **kwargs: Any) -> Any:
    """Call a deployed handle's `.remote()` and meter the client wait."""
    plan = plan_for(name, gpu=gpu)
    extra = {"call": name, **kwargs.pop("cost_extra", {})}
    with span(plan, phase="remote", book=current_book(), ledger=client_ledger(), extra=extra):
        result = fn.remote(*args, **kwargs)
    event = last_event()
    if isinstance(result, dict) and event is not None:
        result = dict(result)
        result["cost"] = event.to_dict()
    return result


def print_last_cost() -> None:
    event = last_event()
    if event is not None:
        print(format_event(event))


def official_summary_text() -> str:
    try:
        from modal import Workspace

        summary = Workspace.from_context().billing.summary()
    except Exception as exc:
        return f"official Modal billing unavailable: {exc}"
    lines = [
        f"official cycle {summary.start.date()} .. {summary.end.date()}",
        f"metered {format_usd(summary.metered_cost)}",
        f"billed  {format_usd(summary.billed_cost)}",
    ]
    for key, value in sorted(summary.metered_cost_breakdown.items(), key=lambda item: -item[1]):
        lines.append(f"  {key}: {format_usd(value)}")
    return "\n".join(lines)


def cost_command(*, official: bool = False) -> int:
    report = ledger_report()
    print(format_trace_tree(report["traces"]))
    if report["event_count"]:
        print(f"ledger {report['ledger_path']}")
        print(f"all-time billed estimate {report['billed_display']}")
        rates = report["rates"]
        print(
            f"per-second cpu {format_per_second(rates['cpu_per_second'])}  "
            f"mem {format_per_second(rates['memory_per_second'])}"
        )
        for gpu, row in rates["gpus"].items():
            print(f"per-second gpu {gpu} {format_per_second(row['usd_per_second'])}  ({row['usd_per_hour']}/h)")
    else:
        print(summarize_events([]))
    if official:
        print(official_summary_text())
    return 0
