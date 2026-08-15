from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Sequence

from .recipes import apply_recipe


@dataclass
class CliCommand:
    action: str
    uris: list[str] = field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = ""
    recipe: str = "sd15"
    model: str = ""
    output: str = "output.png"
    width: int = 0
    height: int = 0
    steps: int = 0
    seed: int = 42
    cfg_scale: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        request = apply_recipe(
            self.recipe,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt or None,
            model=self.model or None,
            width=self.width or None,
            height=self.height or None,
            steps=self.steps or None,
            seed=self.seed,
            cfg_scale=self.cfg_scale or None,
        )
        request.validate()
        return request.to_dict()


def parse_argv(argv: Sequence[str]) -> CliCommand:
    parser = argparse.ArgumentParser(
        prog="sdcpp-modal",
        description="Pull models onto a Modal Volume and run sd-cli remotely.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    pull = sub.add_parser("pull", help="download model URIs onto Modal volume sdcpp-models")
    pull.add_argument("uris", nargs="+", help="hf://, civitai://, or https:// URI")

    sub.add_parser("ls", help="list files already stored on the Modal volume")
    sub.add_parser("probe", help="print remote sd-cli flags (no download)")

    generate = sub.add_parser("generate", help="run txt2img on Modal GPU")
    generate.add_argument("-p", "--prompt", required=True, help="text prompt")
    generate.add_argument("-n", "--negative-prompt", default="", help="negative prompt")
    generate.add_argument("-m", "--model", default="", help="model URI; overrides recipe model")
    generate.add_argument("--recipe", default="sd15", help="bundled recipe (default: sd15)")
    generate.add_argument("-o", "--output", default="output.png", help="local output path")
    generate.add_argument("-W", "--width", type=int, default=0)
    generate.add_argument("-H", "--height", type=int, default=0)
    generate.add_argument("--steps", type=int, default=0)
    generate.add_argument("--cfg-scale", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(list(argv))
    return CliCommand(
        action=args.action,
        uris=list(getattr(args, "uris", []) or []),
        prompt=getattr(args, "prompt", "") or "",
        negative_prompt=getattr(args, "negative_prompt", "") or "",
        recipe=getattr(args, "recipe", "sd15") or "sd15",
        model=getattr(args, "model", "") or "",
        output=getattr(args, "output", "output.png") or "output.png",
        width=int(getattr(args, "width", 0) or 0),
        height=int(getattr(args, "height", 0) or 0),
        steps=int(getattr(args, "steps", 0) or 0),
        seed=int(getattr(args, "seed", 42) or 42),
        cfg_scale=float(getattr(args, "cfg_scale", 0.0) or 0.0),
    )
