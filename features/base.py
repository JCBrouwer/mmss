from abc import abstractmethod
from math import ceil
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

mp.set_sharing_strategy("file_system")


def chunk(list, chunks):
    n = ceil(len(list) / chunks)
    return (list[i : i + n] for i in range(0, len(list), n))


def transpose(list_of_lists):
    return tuple(map(list, zip(*list_of_lists)))


class Feature:
    def __init__(self, dataset, batch_size, num_workers):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_generator = (
            batch
            for batch in DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        )

        self.models = []

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "input_generator"}

    def __setstate__(self, state):
        self.__dict__ = state

    def worker_init_fn(self):
        """Initialize models used for processing the feature on the correct device"""
        rank = mp.current_process()._identity[0]
        device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        for model in self.models:
            model.initialize(device)

    def loader(self):
        """A generator that processes batches in parallel"""
        with mp.Pool(self.num_workers, initializer=self.worker_init_fn) as pool:
            for batch in self.input_generator:
                for fns, embds in pool.imap_unordered(
                    self.process_batch, chunk(batch, self.batch_size / self.num_workers)
                ):
                    yield fns, embds

    def process(self):
        """Returns the processed features"""
        with tqdm(total=self.length, smoothing=0) as progress:
            filenames, embeddings = [], []
            for fns, embds in self.loader():
                filenames += fns
                embeddings += embds
                progress.update(len(fns))
        return np.array(filenames), np.concatenate(embeddings)

    @abstractmethod
    def process_batch(self, batch) -> Tuple[List[str], List[np.array]]:
        """Processes a batch of input data into the feature"""
        return transpose(batch)


if __name__ == "__main__":
    from glob import glob
    from dataloader import Images

    files = glob("/home/hans/datasets/cyphept/cypherpunk/*.jpg")

    feature = Feature(Images(files), batch_size=64, num_workers=8)
    filenames, embeddings = feature.process()
    print(embeddings.shape)

    feature = Feature(Images(files), batch_size=64, num_workers=8)
    filenames, embeddings = [], []
    for fns, embds in feature.loader():
        filenames += fns
        embeddings += embds
    print(np.concatenate(embeddings).shape)

    print("all done")
