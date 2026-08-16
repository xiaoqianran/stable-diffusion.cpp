"""Turn the cost ledger into traces, per-second rates, and job totals."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterable

from .cost import (
    CostEvent,
    PriceBook,
    billed_usd,
    default_plan,
    format_per_second,
    format_usd,
)
from .gpu import ALLOWED_GPU_IDS
from .meter import client_ledger, client_ledger_path


PHASE_ORDER = {"session": 0, "remote": 1, "container": 2, "local": 3}


def event_job_id(event: CostEvent) -> str:
    return event.job_id or str((event.extra or {}).get("job_id") or "")


def _duration_s(event: CostEvent) -> float:
    return event.duration_ms / 1000.0


def breakdown_lines(event: CostEvent) -> list[str]:
    duration_s = _duration_s(event)
    lines = []
    for key, usd in (event.breakdown or {}).items():
        rate = (event.rates or {}).get(key)
        if rate:
            lines.append(f"{key} {format_per_second(rate)} × {duration_s:.3f}s = {format_usd(usd)}")
        else:
            lines.append(f"{key} {format_usd(usd)}")
    return lines


def event_view(event: CostEvent) -> dict[str, Any]:
    duration_s = _duration_s(event)
    per_s = event.usd_per_second or "0"
    return {
        "id": event.id,
        "parent_id": event.parent_id or None,
        "trace_id": event.trace_id,
        "phase": event.phase,
        "name": event.name,
        "started_at": event.started_at,
        "duration_ms": event.duration_ms,
        "duration_s": round(duration_s, 3),
        "usd": event.usd,
        "usd_per_second": per_s,
        "line": f"{format_per_second(per_s)} × {duration_s:.3f}s = {format_usd(event.usd)}",
        "breakdown": event.breakdown,
        "breakdown_lines": breakdown_lines(event),
        "rates": event.rates,
        "gpu": event.gpu,
        "cpu_cores": event.cpu_cores,
        "memory_gib": event.memory_gib,
        "job_id": event_job_id(event) or None,
        "image_id": event.image_id or event.extra.get("image_id") or None,
        "recipe": event.recipe or event.extra.get("recipe") or None,
        "notes": event.notes,
        "rates_source": event.rates_source,
        "call": (event.extra or {}).get("call") or event.name,
    }


def _chain_order(items: list[CostEvent]) -> list[CostEvent]:
    ids = {item.id for item in items}
    children: dict[str, list[CostEvent]] = defaultdict(list)
    roots: list[CostEvent] = []
    for item in items:
        parent = item.parent_id
        if parent and parent in ids and parent != item.id:
            children[parent].append(item)
        else:
            roots.append(item)

    ordered: list[CostEvent] = []

    def walk(node: CostEvent) -> None:
        ordered.append(node)
        kids = sorted(
            children.get(node.id, []),
            key=lambda item: (item.started_at, PHASE_ORDER.get(item.phase, 9), item.name),
        )
        for kid in kids:
            walk(kid)

    for root in sorted(roots, key=lambda item: (item.started_at, PHASE_ORDER.get(item.phase, 9), item.name)):
        walk(root)
    return ordered


def _depth(event: CostEvent, by_id: dict[str, CostEvent]) -> int:
    depth = 0
    seen = {event.id}
    parent = event.parent_id
    while parent and parent in by_id and parent not in seen:
        depth += 1
        seen.add(parent)
        parent = by_id[parent].parent_id
    return depth


def annotated_chain(items: Iterable[CostEvent]) -> list[dict[str, Any]]:
    events = list(items)
    by_id = {item.id: item for item in events}
    views = []
    for item in _chain_order(events):
        view = event_view(item)
        view["depth"] = _depth(item, by_id)
        views.append(view)
    return views


def build_traces(events: Iterable[CostEvent]) -> list[dict[str, Any]]:
    groups: dict[str, list[CostEvent]] = defaultdict(list)
    for event in events:
        groups[event.trace_id or event.id].append(event)
    traces = []
    for trace_id, items in groups.items():
        chain = annotated_chain(items)
        job_ids = sorted({event_job_id(item) for item in items if event_job_id(item)})
        recipes = sorted({item.recipe or str((item.extra or {}).get("recipe") or "") for item in items})
        recipes = [item for item in recipes if item]
        traces.append(
            {
                "trace_id": trace_id,
                "started_at": items[0].started_at if items else "",
                "billed_usd": str(billed_usd(items)),
                "event_count": len(items),
                "job_ids": job_ids,
                "recipes": recipes,
                "chain": chain,
            }
        )
    traces.sort(key=lambda row: row["started_at"], reverse=True)
    return traces


def job_totals(events: Iterable[CostEvent]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CostEvent]] = defaultdict(list)
    for event in events:
        job_id = event_job_id(event)
        if job_id:
            grouped[job_id].append(event)
    return {
        job_id: {
            "job_id": job_id,
            "billed_usd": str(billed_usd(items)),
            "event_count": len(items),
            "chain": annotated_chain(items),
        }
        for job_id, items in grouped.items()
    }


def published_rates(book: PriceBook | None = None) -> dict[str, Any]:
    book = book or PriceBook()
    cpu_plan = default_plan("cpu")
    cpu_s, cpu_parts = book.per_second(cpu_plan)
    gpus = {}
    cards = []
    for gpu in ALLOWED_GPU_IDS:
        total, parts = book.per_second(default_plan(gpu, gpu=gpu))
        row = {
            "usd_per_second": str(total),
            "usd_per_hour": str(book.gpu_hour(gpu)),
            "breakdown": {key: str(value) for key, value in parts.items()},
        }
        gpus[gpu] = row
        cards.append(
            {
                "id": gpu,
                "kind": "gpu",
                "label": gpu,
                "usd_per_second": row["usd_per_second"],
                "usd_per_hour": row["usd_per_hour"],
            }
        )
    cards.append(
        {
            "id": "cpu",
            "kind": "cpu",
            "label": "CPU",
            "note": f"{cpu_plan.cpu_cores:g} cores",
            "usd_per_second": str(cpu_parts["cpu"]),
            "usd_per_hour": str(book.cpu_hour() * Decimal(str(cpu_plan.cpu_cores))),
        }
    )
    cards.append(
        {
            "id": "mem",
            "kind": "memory",
            "label": "内存",
            "note": f"{cpu_plan.memory_gib:g} GiB",
            "usd_per_second": str(cpu_parts["memory"]),
            "usd_per_hour": str(book.memory_hour() * Decimal(str(cpu_plan.memory_gib))),
        }
    )
    return {
        "source": book.source,
        "cpu_per_second": str(cpu_parts["cpu"]),
        "memory_per_second": str(cpu_parts["memory"]),
        "cpu_memory_per_second": str(cpu_s),
        "gpus": gpus,
        "cards": cards,
    }


def _billed_window(events: list[CostEvent]) -> dict[str, Any]:
    remote_ms = sum(item.duration_ms for item in events if item.phase == "remote")
    session_ms = sum(item.duration_ms for item in events if item.phase == "session")
    local_ms = sum(item.duration_ms for item in events if item.phase == "local")
    leftover_ms = max(0, session_ms - remote_ms)
    billed_ms = remote_ms + local_ms + leftover_ms
    total = billed_usd(events)
    return {
        "usd": str(total),
        "display": format_usd(total),
        "event_count": len(events),
        "duration_s": round(billed_ms / 1000.0, 3),
        "remote_s": round(remote_ms / 1000.0, 3),
        "session_s": round(session_ms / 1000.0, 3),
    }


def ledger_report(
    events: list[CostEvent] | None = None,
    *,
    book: PriceBook | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    events = list(events if events is not None else client_ledger().read())
    if job_id:
        events = [event for event in events if event_job_id(event) == job_id]
    book = book or PriceBook()
    traces = build_traces(events)
    jobs = job_totals(events)
    billed = _billed_window(events)
    return {
        "ledger_path": str(client_ledger_path()),
        "event_count": len(events),
        "billed_usd": billed["usd"],
        "billed_display": billed["display"],
        "billed": billed,
        "rates": published_rates(book),
        "traces": traces,
        "jobs": jobs,
        "events": [event_view(event) for event in events],
        "job_id": job_id,
    }


def format_trace_tree(traces: list[dict[str, Any]], *, limit: int = 8) -> str:
    if not traces:
        return "cost ledger is empty"
    lines = []
    for trace in traces[:limit]:
        jobs = ",".join(trace["job_ids"]) or "—"
        lines.append(
            f"trace {trace['trace_id']}  {trace['billed_usd']}  jobs {jobs}  {trace['event_count']} calls"
        )
        for event in trace["chain"]:
            mark = "  " * (1 + int(event.get("depth") or 0))
            job = f"  job {event['job_id']}" if event["job_id"] else ""
            img = f"  img {event['image_id']}" if event["image_id"] else ""
            gpu = f"  gpu {event['gpu']}" if event["gpu"] else ""
            lines.append(
                f"{mark}{event['phase']}:{event['name']}  {event['duration_s']:.3f}s  "
                f"{format_usd(event['usd'])}  {format_per_second(event['usd_per_second'] or 0)}"
                f"{gpu}{job}{img}"
            )
            for part in event.get("breakdown_lines") or []:
                lines.append(f"{mark}  {part}")
    return "\n".join(lines)
