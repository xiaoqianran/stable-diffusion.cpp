from __future__ import annotations

import re
from dataclasses import dataclass, field


FLAG_LINE = re.compile(
    r"^\s+"
    r"(?:(-[A-Za-z0-9]),\s+)?"
    r"(--[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"(?:\s+<(string|int|float)>)?"
)


@dataclass(frozen=True)
class Flag:
    name: str
    short: str | None = None
    value_hint: str | None = None


@dataclass
class EngineCapabilities:
    binary: str
    raw_help: str
    flags: dict[str, Flag] = field(default_factory=dict)

    def has_flag(self, name: str) -> bool:
        return name in self.flags

    def flag(self, name: str) -> Flag:
        return self.flags[name]


def discover_engine(help_text: str, binary: str = "sd-cli") -> EngineCapabilities:
    flags: dict[str, Flag] = {}
    for line in help_text.splitlines():
        match = FLAG_LINE.match(line)
        if match is None:
            continue
        short, long_name, hint = match.group(1), match.group(2), match.group(3)
        parsed = Flag(name=long_name, short=short, value_hint=hint)
        flags[long_name] = parsed
        if short:
            flags[short] = parsed
    return EngineCapabilities(binary=binary, raw_help=help_text, flags=flags)
