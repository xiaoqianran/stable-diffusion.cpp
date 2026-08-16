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
  ideogram4/<id>.png
  flux2-klein/<id>.png
  flux2-dev/<id>.png
  z-image-turbo/<id>.png
  sdxl-turbo/<id>.png
  sd2/<id>.png
  sd15/<id>.png
```

`models.json` lists the seven bundled families. Unknown slugs still render on Pages under Custom.

Each sidecar also records the prompt plus run facts used on the Pages card: `duration_ms`, `gpu_name`, `cuda_version`, `torch_version`, and extras such as NVIDIA driver, sd-cli version, Python, Modal GPU type, and the container image.
