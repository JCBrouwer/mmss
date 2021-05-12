from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from models import Artemis, Clip

from features.feature import Feature
from features.data import Images
from features.primitives import ModelFeature, ModelPipelineFeature


@dataclass
class RegistryEntry:
    name: str
    insert_fn: Callable[[List[str], int, int], Feature]
    search_fn: Callable[[], np.ndarray]


def clip_search():
    clip = Clip()
    clip.initialize(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return np.array(clip(input("Text query: ")), dtype=np.float32)


REGISTRY = {
    "clip": RegistryEntry(
        name="clip-image-embedding",
        insert_fn=lambda f, bs, nw: ModelFeature(Clip(), Images(f), batch_size=bs, num_workers=nw),
        search_fn=clip_search,
    ),
    "artemis": RegistryEntry(
        name="artemis-caption-clip-text-embedding",
        insert_fn=lambda f, bs, nw: ModelPipelineFeature([Artemis(), Clip()], Images(f), batch_size=bs, num_workers=nw),
        search_fn=clip_search,
    ),
}


def register(name: str, feature_info: dict):
    REGISTRY[name] = RegistryEntry(**feature_info)


def retrieve(features: List[str]):
    if "all" in features:
        return REGISTRY.values()
    return [REGISTRY[name] for name in features]
