from math import ceil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset


def ensure_shape(im, size):
    if not (im.shape[2] == size and im.shape[3] == size):
        # TODO figure out why this happens sometimes
        im = F.interpolate(im, size=(size, size), align_corners=False, mode="bilinear")
    if im.shape[1] == 1:
        im = torch.cat([im] * 3, axis=1)
    if im.shape[1] == 4:
        im = im[:, :3]
    # assert list(im.shape[1:]) == [3, size, size]
    return im


class ImagesNoop(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_tensor = tv.transforms.functional.to_tensor(Image.open(filename).convert("RGB")).unsqueeze(0)
        return filename, image_tensor


class Images(Dataset):
    def __init__(self, filenames, size=300):
        self.filenames = filenames
        self.size = size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_tensor = tv.transforms.functional.to_tensor(Image.open(filename).convert("RGB")).unsqueeze(0)
        image_tensor = F.interpolate(
            image_tensor, scale_factor=self.size / min(image_tensor.shape[2:]), recompute_scale_factor=False
        )
        image_tensor = tv.transforms.functional.center_crop(image_tensor, self.size)
        return filename, ensure_shape(image_tensor, self.size)


class VideoFrames(Dataset):
    def __init__(self, filenames, size=300, num_frames=8):
        self.filenames = filenames
        self.size = size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        try:
            cap = cv2.VideoCapture(filename)
            assert cap.isOpened(), f"{filename} could not be opened!"

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= self.num_frames:
                frame_selection = np.arange(frame_count)
            else:
                frame_bins = np.linspace(0, frame_count, self.num_frames + 1).astype(np.int32)
                frame_selection = [np.random.randint(frame_bins[i], frame_bins[i + 1]) for i in range(self.num_frames)]

            frames = []
            for frame_index in frame_selection:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame) / 255
                frame = frame.permute(2, 0, 1).unsqueeze(0)
                frame = F.interpolate(
                    frame, scale_factor=self.size / min(frame.shape[2:]), recompute_scale_factor=False
                )
                frame = tv.transforms.CenterCrop(self.size)(frame)
                frame = ensure_shape(frames, self.size)
                frames.append(frame)

            return [filename] * len(frames), frames
        except Exception as e:
            print("\n\nERROR: Processing", filename, "failed!\n", e, "\n")
            return filename, None
