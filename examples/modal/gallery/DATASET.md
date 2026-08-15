---
license: mit
task_categories:
  - text-to-image
tags:
  - stable-diffusion.cpp
  - sd-cli
  - gallery
pretty_name: stable-diffusion.cpp gallery
---

# stable-diffusion.cpp gallery

Public dataset of images generated with [stable-diffusion.cpp](https://github.com/xiaoqianran/stable-diffusion.cpp).

Layout is one folder per model family. Future models do not need a code change: publish with `--model-id <slug>` and a new `images/<slug>/` folder appears.

```
images/
  sd15/<id>.png
  sd15/<id>.json
  sdxl/<id>.png
  flux/<id>.png
  wan/<id>.png
```

`models.json` lists known families (SD, FLUX, Qwen, Wan, ...). Unknown slugs still render on Pages under Custom.
