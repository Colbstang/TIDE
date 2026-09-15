"""Shared paths and the pinned MedCPT loader for TIDE."""
from __future__ import annotations
from functools import lru_cache
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EMB = os.environ.get("TIDE_EMB", f"{ROOT}/data/embeddings")
RAW = f"{ROOT}/data/raw"
OUT = os.environ.get("TIDE_OUT", f"{ROOT}/out")
os.makedirs(OUT, exist_ok=True)

with open(f"{ROOT}/provenance.json", encoding="utf-8") as f:
    PROVENANCE = json.load(f)

# Direction palette, colorblind-safe (blue, orange, grey), used across all figures.
# blue = diverging (declining trend), orange = converging (rising trend), grey = flat.
DIR_BLUE = "#2171b5"
DIR_GREY = "#7f7f7f"
DIR_ORANGE = "#e6550d"
BLUE_FAMILY = ["#08519c", "#3182bd", "#6baed6"]      # multiple receding terms
ORANGE_FAMILY = ["#a63603", "#e6550d", "#fd8d3c"]    # multiple rising terms
GREY_FAMILY = ["#525252", "#737373", "#969696", "#bdbdbd", "#d9d9d9"]


def nrm(v):
    return v / (np.linalg.norm(v) + 1e-9)


@lru_cache(maxsize=1)
def load_medcpt():
    """Load the two immutable MedCPT revisions on CPU.

    Set ``TIDE_OFFLINE=1`` to require an already-populated Hugging Face cache.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(int(os.environ.get("TIDE_THREADS", "8")))
    offline = os.environ.get("TIDE_OFFLINE", "").lower() in {"1", "true", "yes"}

    def load(spec):
        kwargs = {"revision": spec["revision"], "local_files_only": offline}
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], **kwargs)
        model = AutoModel.from_pretrained(spec["id"], **kwargs).eval()
        return tokenizer, model

    qtok, qmod = load(PROVENANCE["models"]["query_encoder"])
    atok, amod = load(PROVENANCE["models"]["article_encoder"])
    return qtok, qmod, atok, amod
