from dataclasses import dataclass
from typing import Callable, List

from features.data import Images
from features.feature import Feature
from features.primitives import ModelFeature, ModelPipelineFeature
from models import Artemis, Clip, SearchableModel
from models.hist import Histogram


@dataclass
class RegistryEntry:

    name: str
    insert_fn: Callable[[List[str], int, int, SearchableModel], Feature]
    end_search: SearchableModel


REGISTRY = {
    "clip": RegistryEntry(
        name="clip-image-embedding",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, Images(f), batch_size=bs, num_workers=nw),
        end_search=Clip()
    ),
    "artemis": RegistryEntry(
        name="artemis-caption-clip-text-embedding",
        insert_fn=lambda f, bs, nw, em: ModelPipelineFeature([Artemis(), em], Images(f), batch_size=bs, num_workers=nw),
        end_search=Clip()
    ),
    "histogram": RegistryEntry(
        name="histogram-image-search",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, Images(f), batch_size=bs, num_workers=nw),
        end_search=Histogram()
    )
}


def register(name: str, feature_info: dict):
    REGISTRY[name] = RegistryEntry(**feature_info)


def retrieve(features: List[str]):
    if "all" in features:
        return REGISTRY.values()
    return [REGISTRY[name] for name in features]
