import gc
from collections import defaultdict
from typing import Union

import torch
import torchvision as tv
from PIL.Image import Image
from torch.tensor import Tensor

from processors.base import Processor


class Yolo(Processor):
    def __init__(self,):
        self.model = None
        self.device = None

        # Max number of classes per image that we should identify
        self.output_size = 10

        # Min confidence for class detection
        self.min_confidence = 0.6

    def initialize(self, device):
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, device=device).eval()
        if torch.cuda.is_available():
            self.model = self.model.half()

    def __call__(self, inputs: Union[Image, Tensor]):
        img = tv.transforms.functional.to_pil_image(inputs.squeeze())
        results = self.model(img)

        # If this fails then yolo might not return result for some inputs and call needs adjustment for batching
        assert len(results) == len(inputs)

        outputs = []
        names = results.names

        # Allow for batch results distributed per image
        for pred in results.pred:
            # If there even is any pred
            if pred is not None and len(pred) > 0:
                # Default dict such that there are no key errors/checks required
                name_dict_img = defaultdict(float)
                # Pred consists of box params and ends with conf, cls so *_ ignored (still tensors)
                for *_, conf, cls in pred:
                    c_name = names[int(cls.item())]
                    # Check if the confidence is high enough
                    if float(conf.item()) > self.min_confidence:
                        # Add to total confidence for this label
                        name_dict_img[c_name] += float(conf.item())
                sorted_res = [pair[0] for pair in sorted(name_dict_img.items(), key=lambda item: item[1], reverse=True)]
                if len(sorted_res) > self.output_size:
                    sorted_res = sorted_res[: self.output_size]
                outputs.append(" ".join(sorted_res))

            else:
                outputs.append("")

        return outputs
