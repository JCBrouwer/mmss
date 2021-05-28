from collections import defaultdict
from typing import List, Union
from PIL.Image import Image
from torch.tensor import Tensor
import torch
from models.model import Model


class YoloClasses(Model):
    def __init__(self, ):
        self.model = None
        self.device = None

        # Max number of classes per image that we should identify
        self.output_size = 10

        # Min confidence for class detection
        self.min_confidence = 0.6

    def initialize(self, device):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def __call__(self, inputs: List[Union[Image, Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        results = self.model(inputs)
        # If this fails then yolov5s might not return result for some inputs and call needs adjustment for batching
        assert len(results) == len(inputs)

        outputs = [[] for _ in range(len(inputs))]
        names = results.names

        # Allow for batch results distributed per image
        for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
            # If there even is any pred
            if pred is not None:
                # Default dict such that there are no key errors/checks required
                name_dict_img = defaultdict(float)
                # Pred consists of box params and ends with conf, cls so *_ ignored (still tensors)
                for *_, conf, cls in pred:
                    c_name = names[int(cls)]
                    # Check if the confidence is high enough
                    if float(conf) > self.min_confidence:
                        # Add to total confidence for this label
                        name_dict_img[c_name] += float(conf)
                sorted_res = [pair[0] for pair in sorted(name_dict_img.items(), key=lambda item: item[1], reverse=True)]
                if len(sorted_res) > self.output_size:
                    sorted_res = sorted_res[:self.output_size]
                outputs[i] = sorted_res

        return outputs
