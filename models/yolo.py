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

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie