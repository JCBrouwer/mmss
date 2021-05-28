from dataclasses import dataclass
from typing import Callable, List

from models import Artemis, Clip, SearchableModel, YoloClasses  # , Histogram
from models.keypoint import BRISK, ORB, SIFT

from features.data import Images, ImagesNoop
from features.feature import Feature
from features.primitives import ModelFeature, ModelPipelineFeature


@dataclass
class RegistryEntry:
    name: str
    insert_fn: Callable[[List[str], int, int, SearchableModel], Feature]
    search_model: SearchableModel


REGISTRY = {
    "clip": RegistryEntry(
        name="clip-image-embedding",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, Images(f), batch_size=bs, num_workers=nw),
        search_model=Clip(),
    ),
    "artemis": RegistryEntry(
        name="artemis-caption-clip-text-embedding",
        insert_fn=lambda f, bs, nw, em: ModelPipelineFeature([Artemis(), em], Images(f), batch_size=bs, num_workers=nw),
        search_model=Clip(),
    ),
    # "histogram": RegistryEntry(
    #     name="color-histogram",
    #     insert_fn=lambda f, bs, nw, em: ModelFeature(em, Images(f), batch_size=bs, num_workers=nw),
    #     search_model=Histogram(),
    # ),
    "sift": RegistryEntry(
        name="sift",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, ImagesNoop(f), batch_size=bs, num_workers=nw),
        search_model=SIFT(),
    ),
    "orb": RegistryEntry(
        name="orb",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, ImagesNoop(f), batch_size=bs, num_workers=nw),
        search_model=ORB(),
    ),
    "brisk": RegistryEntry(
        name="brisk",
        insert_fn=lambda f, bs, nw, em: ModelFeature(em, ImagesNoop(f), batch_size=bs, num_workers=nw),
        search_model=BRISK(),
    ),
    "yolo": RegistryEntry(
        name="yolo-classes-clip-text-embedding",
        insert_fn=lambda f, bs, nw, em: ModelPipelineFeature(
            [YoloClasses(), em], ImagesNoop(f), batch_size=bs, num_workers=nw
        ),
        search_model=Clip(),
    ),
}


def register(name: str, feature_info: dict):
    REGISTRY[name] = RegistryEntry(**feature_info)


def retrieve(features: List[str]):
    if "all" in features:
        return REGISTRY.values()
    return [REGISTRY[name] for name in features]
