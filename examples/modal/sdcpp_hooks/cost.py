from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Mapping


# Snapshot of `modal billing rates` for workspace pythonmoive (2026-08-15).
# Live rates replace this when Modal is reachable; tests use this table.
FALLBACK_RATES: dict[str, str] = {
    "cpu_hour_cost": "0.04730",
    "cpu_hour_cost_sandbox": "0.141900",
    "gpu_hour_cost_a100_40gb": "2.10000",
    "gpu_hour_cost_a100_80gb": "2.50000",
    "gpu_hour_cost_a10g": "1.10000",
    "gpu_hour_cost_b200": "6.25000",
    "gpu_hour_cost_b300": "7.10000",
    "gpu_hour_cost_h100": "3.95000",
    "gpu_hour_cost_h200": "4.54000",
    "gpu_hour_cost_l4": "0.80000",
    "gpu_hour_cost_l40s": "1.95000",
    "gpu_hour_cost_rtx6000": "3.03000",
    "gpu_hour_cost_t4": "0.59000",
    "mem_gib_hour_cost": "0.00800",
    "mem_gib_hour_cost_sandbox": "0.024000",
    "volume_storage_gib_month_cost": "0.09000",
}

DEFAULT_CPU_CORES = 0.125
DEFAULT_MEMORY_GIB = 1.0
USD_QUANT = Decimal("0.000001")

GPU_ALIASES = {
    "A10": "A10G",
    "A100": "A100-40GB",
    "A100:40GB": "A100-40GB",
    "A100-40GB": "A100-40GB",
    "A100:80GB": "A100-80GB",
    "A100-80GB": "A100-80GB",
    "L40S": "L40S",
    "RTX6000": "RTX6000",
    "RTX-PRO-6000": "RTX6000",
    "RTXPRO6000": "RTX6000",
    "PRO-6000": "RTX6000",
    "PRO6000": "RTX6000",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def format_usd(value: Decimal | float | str) -> str:
    quantized = _dec(value).quantize(USD_QUANT, rounding=ROUND_HALF_UP)
    return f"${quantized}"


def gpu_rate_key(gpu: str) -> str:
    text = (gpu or "").strip().upper().replace(" ", "-")
    text = GPU_ALIASES.get(text, text)
    return "gpu_hour_cost_" + text.lower().replace("-", "_")


@dataclass(frozen=True)
class ResourcePlan:
    name: str
    gpu: str | None = None
    cpu_cores: float = DEFAULT_CPU_CORES
    memory_gib: float = DEFAULT_MEMORY_GIB
    assumed_defaults: bool = True
    notes: str = ""


@dataclass
class CostEvent:
    id: str
    name: str
    phase: str
    started_at: str
    duration_ms: int
    usd: str
    breakdown: dict[str, str] = field(default_factory=dict)
    gpu: str | None = None
    cpu_cores: float = DEFAULT_CPU_CORES
    memory_gib: float = DEFAULT_MEMORY_GIB
    rates_source: str = "fallback"
    trace_id: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostEvent:
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        known.setdefault("breakdown", {})
        known.setdefault("extra", {})
        return cls(**known)


class PriceBook:
    def __init__(self, rates: Mapping[str, Any] | None = None, source: str = "fallback"):
        raw = dict(FALLBACK_RATES)
        if rates:
            raw.update({str(key): str(value) for key, value in rates.items()})
        self.rates = {key: _dec(value) for key, value in raw.items()}
        self.source = source

    def cpu_hour(self) -> Decimal:
        return self.rates["cpu_hour_cost"]

    def memory_hour(self) -> Decimal:
        return self.rates["mem_gib_hour_cost"]

    def volume_month(self) -> Decimal:
        return self.rates["volume_storage_gib_month_cost"]

    def gpu_hour(self, gpu: str) -> Decimal:
        key = gpu_rate_key(gpu)
        if key not in self.rates:
            raise KeyError(f"no Modal rate for GPU {gpu!r} ({key})")
        return self.rates[key]

    def estimate(self, plan: ResourcePlan, duration_s: float) -> tuple[Decimal, dict[str, Decimal]]:
        hours = _dec(max(duration_s, 0.0)) / Decimal(3600)
        breakdown = {
            "cpu": (self.cpu_hour() * _dec(plan.cpu_cores) * hours),
            "memory": (self.memory_hour() * _dec(plan.memory_gib) * hours),
        }
        if plan.gpu:
            breakdown["gpu"] = self.gpu_hour(plan.gpu) * hours
        total = sum(breakdown.values(), Decimal("0"))
        return total, breakdown

    def estimate_volume_month(self, bytes_used: int) -> Decimal:
        gib = _dec(max(bytes_used, 0)) / Decimal(1024**3)
        return self.volume_month() * gib


def default_plan(name: str, *, gpu: str | None = None, notes: str = "") -> ResourcePlan:
    return ResourcePlan(name=name, gpu=gpu or None, notes=notes)


def make_event(
    *,
    event_id: str,
    plan: ResourcePlan,
    duration_ms: int,
    book: PriceBook,
    phase: str,
    trace_id: str = "",
    started_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> CostEvent:
    total, breakdown = book.estimate(plan, duration_ms / 1000.0)
    return CostEvent(
        id=event_id,
        name=plan.name,
        phase=phase,
        started_at=started_at or utc_now_iso(),
        duration_ms=max(duration_ms, 0),
        usd=str(total.quantize(USD_QUANT, rounding=ROUND_HALF_UP)),
        breakdown={key: str(value.quantize(USD_QUANT, rounding=ROUND_HALF_UP)) for key, value in breakdown.items()},
        gpu=plan.gpu,
        cpu_cores=plan.cpu_cores,
        memory_gib=plan.memory_gib,
        rates_source=book.source,
        trace_id=trace_id,
        notes=plan.notes,
        extra=extra or {},
    )


def _trace_billed_usd(items: list[CostEvent]) -> Decimal:
    remotes = [item for item in items if item.phase == "remote"]
    sessions = [item for item in items if item.phase == "session"]
    total = sum((_dec(item.usd) for item in remotes), Decimal("0"))
    remote_ms = sum(item.duration_ms for item in remotes)
    session_ms = sum(item.duration_ms for item in sessions)
    overhead_ms = max(0, session_ms - remote_ms)
    if overhead_ms:
        extra, _ = PriceBook().estimate(default_plan("session-overhead"), overhead_ms / 1000.0)
        total += extra
    return total


def billed_usd(events: Iterable[CostEvent]) -> Decimal:
    """Price remotes plus leftover session time as CPU.

    Session and remote overlap for ephemeral `app.run()` + `.remote()`.
    Summing both would double-count the worker window. Events are grouped
    by `trace_id` so many CLI invocations do not mix their overhead.
    """
    groups: dict[str, list[CostEvent]] = {}
    for item in events:
        groups.setdefault(item.trace_id or item.id, []).append(item)
    total = sum((_trace_billed_usd(group) for group in groups.values()), Decimal("0"))
    return total.quantize(USD_QUANT, rounding=ROUND_HALF_UP)


def format_event(event: CostEvent) -> str:
    seconds = event.duration_ms / 1000.0
    parts = [f"cost {event.phase}:{event.name} {seconds:.2f}s {format_usd(event.usd)}"]
    if event.gpu:
        parts.append(f"gpu {event.gpu}")
    parts.append(f"cpu {event.cpu_cores:g}")
    parts.append(f"mem {event.memory_gib:g}GiB")
    if event.rates_source:
        parts.append(event.rates_source)
    return " · ".join(parts)
