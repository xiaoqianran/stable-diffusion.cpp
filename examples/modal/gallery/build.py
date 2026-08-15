#!/usr/bin/env python3
"""Build a paginated static gallery from the Hugging Face dataset checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdcpp_hooks.gallery import build_site, load_families  # noqa: E402
from sdcpp_hooks.hf_dataset import DEFAULT_DATASET  # noqa: E402


def download_dataset(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_DATASET, help="HF dataset repo id")
    parser.add_argument("--dataset-dir", default="", help="local dataset checkout; downloads --repo if omitted")
    parser.add_argument("--out", default=str(HERE / "site"), help="static site output directory")
    parser.add_argument("--per-page", type=int, default=12)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else HERE / ".dataset"
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        download_dataset(args.repo, dataset_dir)

    stats = build_site(
        dataset_dir,
        Path(args.out),
        per_page=max(1, args.per_page),
        families=load_families(),
    )
    print(f"wrote {args.out} images={stats['images']} models={stats['models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
