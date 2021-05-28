from typing import List, Union
from PIL.Image import Image
from torch.tensor import Tensor
import cv2 as cv
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from processors.base import SearchProcessor


class KeyPointMatching(SearchProcessor):
    def __init__(self, type, n_keypoints):
        self.n_keypoints = n_keypoints
        self.device = "cpu"
        self.type = type

        if self.type == "sift":
            self.output_size = 128
        elif self.type == "orb":
            self.output_size = 32
        elif self.type == "brisk":
            self.output_size = 64

    def initialize(self, device):
        if self.type == "sift":
            self.model = cv.SIFT_create(nfeatures=self.n_keypoints)
        elif self.type == "orb":
            self.model = cv.ORB_create(nfeatures=self.n_keypoints)
        elif self.type == "brisk":
            self.model = cv.BRISK_create()

    def __call__(self, inputs: List[Union[Image, Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = []
        for img_or_tensor in inputs:
            if isinstance(img_or_tensor, torch.Tensor):
                img_or_tensor = img_or_tensor.squeeze().permute(1, 2, 0)
            img = np.array(img_or_tensor)
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype("uint8")

            _, descriptors = self.model.detectAndCompute(img, None)

            if descriptors is None:
                outputs.append(np.zeros((self.n_keypoints, self.output_size)))
                continue

            if self.type == "sift":
                # use RootSIFT https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
                descriptors = np.sqrt(descriptors / np.abs(descriptors).max(axis=1)[:, None])

            if self.type == "brisk" and self.n_keypoints != 0:
                descriptors = descriptors[: self.n_keypoints]

            outputs.append(descriptors)
        return outputs

    def search(self, query: List[Union[Image, Tensor]]):
        return np.concatenate(self(query), axis=0).astype(np.float32)


class SIFT(KeyPointMatching):
    def __init__(self, n_keypoints=1000):
        super().__init__("sift", n_keypoints)


class ORB(KeyPointMatching):
    def __init__(self, n_keypoints=1000):
        super().__init__("orb", n_keypoints)


class BRISK(KeyPointMatching):
    def __init__(self, n_keypoints=1000):
        super().__init__("brisk", n_keypoints)

