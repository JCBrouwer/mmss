from typing import List, Union
from PIL.Image import Image
from torch.tensor import Tensor
import cv2 as cv
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from models.model import SearchableModel


class Sift(SearchableModel):
    def __init__(self, n_keypoints=150):
        self.model = None
        self.n_keypoints = n_keypoints
        self.output_size = 128
        self.device = None

    def initialize(self, device):
        self.device = device
        self.model = cv.SIFT_create(nfeatures=self.n_keypoints)

    def to_grayscale(self, image):
        # Converts image to 8-bit grayscale
        imgGray = image.convert("L")
        return np.array(imgGray)

    def __call__(self, inputs: List[Union[Image, Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = []
        for img_or_tensor in inputs:
            if isinstance(img_or_tensor, Tensor):
                img = self.to_grayscale(to_pil_image(img_or_tensor.squeeze()))
            else:
                img = self.to_grayscale(img_or_tensor)

            _, descriptors = self.model.detectAndCompute(img, None)

            if descriptors is None:
                outputs.append(np.zeros((self.n_keypoints, 128)))
                continue

            # use RootSIFT https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
            descriptors = np.sqrt(descriptors / np.abs(descriptors).max(axis=1)[:, None])
            outputs.append(descriptors)
        return outputs

    def search(self, query: List[Union[Image, Tensor]]):
        return np.concatenate(self(query), axis=0)
