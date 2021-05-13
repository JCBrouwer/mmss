from typing import List, Union
from PIL.Image import Image
from torch.tensor import Tensor
import cv2 as cv
import torch
import numpy as np
import torchvision.transforms as transforms
from models.model import Model


class Sift(Model):
    def __init__(self, n_keypoints=150):
        self.model = None
        self.n_keypoints = n_keypoints
        self.output_size = self.n_keypoints * 128
        self.device = None

    def initialize(self, device):
        self.device = device
        self.model = cv.SIFT_create(nfeatures=self.n_keypoints)

    def __call__(self, inputs: List[Union[Image, Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = []
        for img_or_tensor in inputs:
            if isinstance(img_or_tensor, Tensor):
                img = self.convertToGrayscale(transforms.toPillImage()(img_or_tensor))
            else:
                img = self.convertToGrayscale(img_or_tensor)

            _, output = self.model.detectAndCompute(img)
            outputs.append(torch.tensor(output.flatten()))
        return torch.cat(outputs).detach().cpu()

    def convertToGrayscale(self, image):
        # Converts image to 8-bit grayscale
        imgGray = image.convert('L')
        return np.array(imgGray)
