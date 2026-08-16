from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Sequence

from .recipes import apply_recipe, recipe_uris


def parse_extra_cli(argv: Sequence[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    tokens = [token for token in argv if token != "--"]
    index = 0
    while index < len(tokens):
        name = tokens[index]
        if not name.startswith("-"):
            raise SystemExit(f"unexpected extra argument: {name}")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            extra[name] = tokens[index + 1]
            index += 2
            continue
        extra[name] = True
        index += 1
    return extra


@dataclass
class CliCommand:
    action: str
    uris: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = ""
    recipe: str = "sd15"
    model: str = ""
    diffusion_model: str = ""
    uncond_diffusion_model: str = ""
    vae: str = ""
    clip_l: str = ""
    clip_g: str = ""
    clip_vision: str = ""
    t5xxl: str = ""
    llm: str = ""
    llm_vision: str = ""
    lora_dir: str = ""
    init_image: str = ""
    control_net: str = ""
    taesd: str = ""
    upscale_model: str = ""
    output: str = "output.png"
    width: int = 0
    height: int = 0
    steps: int = 0
    seed: int = 42
    cfg_scale: float = 0.0
    sampling_method: str = ""
    scheduler: str = ""
    strength: float | None = None
    batch_count: int = 0
    extra_cli: dict[str, Any] = field(default_factory=dict)
    publish: bool = False
    model_id: str = ""
    image: str = ""
    official: bool = False
    quant: str = "q8_0"

    def to_payload(self) -> dict[str, Any]:
        request = apply_recipe(
            self.recipe,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt or None,
            model=self.model or None,
            diffusion_model=self.diffusion_model or None,
            uncond_diffusion_model=self.uncond_diffusion_model or None,
            vae=self.vae or None,
            clip_l=self.clip_l or None,
            clip_g=self.clip_g or None,
            clip_vision=self.clip_vision or None,
            t5xxl=self.t5xxl or None,
            llm=self.llm or None,
            llm_vision=self.llm_vision or None,
            lora_dir=self.lora_dir or None,
            init_image=self.init_image or None,
            control_net=self.control_net or None,
            taesd=self.taesd or None,
            upscale_model=self.upscale_model or None,
            width=self.width or None,
            height=self.height or None,
            steps=self.steps or None,
            seed=self.seed,
            cfg_scale=self.cfg_scale or None,
            sampling_method=self.sampling_method or None,
            scheduler=self.scheduler or None,
            strength=self.strength,
            batch_count=self.batch_count or None,
            extra_cli=self.extra_cli or None,
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
    pull.add_argument("--recipe", default="", help="also pull files from a bundled recipe")
    pull.add_argument("uris", nargs="*", help="hf://, civitai://, or https:// URI")

    put = sub.add_parser("put", help="upload a local file onto volume sdcpp-models")
    put.add_argument("files", nargs="+", help="local file path (small files; use pull for weights)")

    sub.add_parser("ls", help="list files already stored on the Modal volume")
    sub.add_parser("probe", help="print remote sd-cli flags (no download)")

    generate = sub.add_parser("generate", help="run sd-cli on Modal GPU")
    generate.add_argument("-p", "--prompt", required=True, help="text prompt")
    generate.add_argument("-n", "--negative-prompt", default="", help="negative prompt")
    generate.add_argument("-m", "--model", default="", help="model URI; overrides recipe model")
    generate.add_argument("--diffusion-model", default="", help="standalone diffusion model URI")
    generate.add_argument(
        "--uncond-diffusion-model",
        dest="uncond_diffusion_model",
        default="",
        help="unconditional diffusion model URI (Ideogram4 CFG)",
    )
    generate.add_argument("--vae", default="", help="VAE URI")
    generate.add_argument("--clip-l", dest="clip_l", default="")
    generate.add_argument("--clip-g", dest="clip_g", default="")
    generate.add_argument("--clip-vision", dest="clip_vision", default="")
    generate.add_argument("--t5xxl", default="")
    generate.add_argument("--llm", default="")
    generate.add_argument("--llm-vision", dest="llm_vision", default="")
    generate.add_argument("--lora-model-dir", dest="lora_dir", default="")
    generate.add_argument("-i", "--init-img", "--init-image", dest="init_image", default="")
    generate.add_argument("--control-net", dest="control_net", default="")
    generate.add_argument("--taesd", default="")
    generate.add_argument("--upscale-model", dest="upscale_model", default="")
    generate.add_argument("--recipe", default="sd15", help="bundled recipe (default: sd15)")
    generate.add_argument("-o", "--output", default="output.png", help="local output path")
    generate.add_argument("-W", "--width", type=int, default=0)
    generate.add_argument("-H", "--height", type=int, default=0)
    generate.add_argument("--steps", type=int, default=0)
    generate.add_argument("--cfg-scale", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--sampling-method", default="")
    generate.add_argument("--scheduler", default="")
    generate.add_argument("--strength", type=float, default=None)
    generate.add_argument("-b", "--batch-count", dest="batch_count", type=int, default=0)
    generate.add_argument("--publish", action="store_true", help="upload the PNG to the HF gallery dataset")
    generate.add_argument("--model-id", default="", help="gallery model folder (default: recipe)")

    publish = sub.add_parser("publish", help="upload a local PNG to the HF gallery dataset")
    publish.add_argument("image", help="local PNG path")
    publish.add_argument("-p", "--prompt", default="")
    publish.add_argument("-n", "--negative-prompt", default="")
    publish.add_argument("--recipe", default="sd15")
    publish.add_argument("--model-id", default="", help="gallery model folder (default: recipe)")
    publish.add_argument("--seed", type=int, default=42)
    publish.add_argument("--steps", type=int, default=0)
    publish.add_argument("-W", "--width", type=int, default=0)
    publish.add_argument("-H", "--height", type=int, default=0)
    publish.add_argument("--cfg-scale", type=float, default=0.0)

    convert = sub.add_parser("convert", help="convert Ideogram4 fp8 weights to GGUF on Modal")
    convert.add_argument("--recipe", default="ideogram4", help="recipe to convert (only ideogram4)")
    convert.add_argument("--quant", default="q8_0", help="gguf quant (default: q8_0)")

    cost = sub.add_parser("cost", help="show local Modal cost ledger (and optional official workspace summary)")
    cost.add_argument(
        "--official",
        action="store_true",
        help="also print modal.Workspace.billing.summary for this month",
    )

    args, unknown = parser.parse_known_args(list(argv))
    if unknown and args.action != "generate":
        parser.error("unrecognized arguments: " + " ".join(unknown))

    uris = list(getattr(args, "uris", []) or [])
    recipe = getattr(args, "recipe", "") or ""
    if args.action == "pull":
        if recipe:
            try:
                uris = recipe_uris(recipe) + uris
            except KeyError as exc:
                parser.error(str(exc))
        if not uris:
            parser.error("pull requires at least one URI or --recipe")

    return CliCommand(
        action=args.action,
        uris=uris,
        files=list(getattr(args, "files", []) or []),
        prompt=getattr(args, "prompt", "") or "",
        negative_prompt=getattr(args, "negative_prompt", "") or "",
        recipe=recipe or "sd15",
        model=getattr(args, "model", "") or "",
        diffusion_model=getattr(args, "diffusion_model", "") or "",
        uncond_diffusion_model=getattr(args, "uncond_diffusion_model", "") or "",
        vae=getattr(args, "vae", "") or "",
        clip_l=getattr(args, "clip_l", "") or "",
        clip_g=getattr(args, "clip_g", "") or "",
        clip_vision=getattr(args, "clip_vision", "") or "",
        t5xxl=getattr(args, "t5xxl", "") or "",
        llm=getattr(args, "llm", "") or "",
        llm_vision=getattr(args, "llm_vision", "") or "",
        lora_dir=getattr(args, "lora_dir", "") or "",
        init_image=getattr(args, "init_image", "") or "",
        control_net=getattr(args, "control_net", "") or "",
        taesd=getattr(args, "taesd", "") or "",
        upscale_model=getattr(args, "upscale_model", "") or "",
        output=getattr(args, "output", "output.png") or "output.png",
        width=int(getattr(args, "width", 0) or 0),
        height=int(getattr(args, "height", 0) or 0),
        steps=int(getattr(args, "steps", 0) or 0),
        seed=int(getattr(args, "seed", 42) or 42),
        cfg_scale=float(getattr(args, "cfg_scale", 0.0) or 0.0),
        sampling_method=getattr(args, "sampling_method", "") or "",
        scheduler=getattr(args, "scheduler", "") or "",
        strength=getattr(args, "strength", None),
        batch_count=int(getattr(args, "batch_count", 0) or 0),
        extra_cli=parse_extra_cli(unknown) if args.action == "generate" else {},
        publish=bool(getattr(args, "publish", False)),
        model_id=getattr(args, "model_id", "") or "",
        image=getattr(args, "image", "") or "",
        official=bool(getattr(args, "official", False)),
        quant=getattr(args, "quant", "q8_0") or "q8_0",
    )
