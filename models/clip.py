from typing import List, Union
from PIL.Image import Image
import torch
import torch.nn.functional as F
from torch.tensor import Tensor
import torchvision as tv

import clip
from clip.simple_tokenizer import SimpleTokenizer

from .base import Model

CLIP_N_PIX = 224


class Clip(Model):
    def __init__(self, backbone="ViT-B/32"):
        self.backbone = backbone
        self.model = None

        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]

    def initialize(self, device):
        self.device = device
        self.model, _ = clip.load(self.backbone, device=device)

    def preprocess(self, tens):
        tens = F.interpolate(tens, size=CLIP_N_PIX, align_corners=False, mode="bilinear")
        tens = tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(tens)
        return tens

    def tokenize(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [[self.sot_token] + self.tokenizer.encode(text) + [self.eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            n_tokens = min(context_length, len(tokens))
            if n_tokens == context_length:
                print(f"WARNING: '{texts[i]}' too long for CLIP context length (77)")
            result[i, :n_tokens] = torch.tensor(tokens)[:n_tokens]
        return result

    def __call__(self, inputs: List[Union[Image, str, Tensor]]):
        with torch.no_grad():
            if not isinstance(inputs, list):
                inputs = [inputs]
            outputs = []
            for img_or_text in inputs:
                if isinstance(img_or_text, str):
                    text = self.tokenize(img_or_text).to(self.device)
                    outputs.append(self.model.encode_text(text))
                else:
                    if isinstance(img_or_text, Image):
                        img_or_text = tv.transforms.ToTensor()(img_or_text)
                    img = self.preprocess(img_or_text)
                    outputs.append(self.model.encode_image(img))
        return torch.cat(outputs).cpu()

