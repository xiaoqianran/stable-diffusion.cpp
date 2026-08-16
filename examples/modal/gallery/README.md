# Generation gallery

Images live on Hugging Face dataset
[`seachen/stable-diffusion-cpp-gallery`](https://huggingface.co/datasets/seachen/stable-diffusion-cpp-gallery).

Each model family is a folder:

```
images/<model-id>/<id>.png
images/<model-id>/<id>.json
```

`models.json` lists the seven bundled Modal recipes. A future model only
needs `--model-id its-slug`.

Build the paginated site locally:

```bash
python3 examples/modal/gallery/build.py --dataset-dir /path/to/checkout --out /tmp/gallery-site
```

GitHub Actions deploys that site to Pages.
