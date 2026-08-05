# py/loader.py
import json
import os
from glob import glob

import torch
from safetensors import safe_open
from types import SimpleNamespace

from model import Qwen2


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return SimpleNamespace(**c)


def load_model(path, device="cuda", dtype=torch.float16):
    cfg = load_config(path)
    model = Qwen2(cfg, device=device, dtype=dtype)

    sd = model.state_dict()
    seen = set()

    for f in glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                # HF prefixes everything with "model."; lm_head is absent
                # entirely because embeddings are tied.
                key = name[len("model."):] if name.startswith("model.") else name
                if key not in sd:
                    print(f"[loader] skipping unmapped {name}")
                    continue
                w = sf.get_tensor(name)
                assert sd[key].shape == w.shape, f"{key}: {sd[key].shape} vs {w.shape}"
                sd[key].copy_(w.to(dtype))
                seen.add(key)

    missing = set(sd.keys()) - seen
    assert not missing, f"missing weights: {sorted(missing)[:8]}"

    model.eval()
    return model, cfg
