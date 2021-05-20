from typing import List, Union
from PIL.Image import Image
from torch.tensor import Tensor
import torch
from models.model import Model


class Yolo(Model):
    def __init__(self, n_keypoints=150):
        self.model = None
        self.n_keypoints = n_keypoints
        self.output_size = self.n_keypoints * 128
        self.device = None

    def initialize(self, device):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def __call__(self, inputs: List[Union[Image, Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        results = self.model(inputs)
        return results
