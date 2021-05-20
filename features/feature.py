import multiprocessing as mp
import threading
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from math import ceil
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def chunk(list, chunks):
    n = ceil(len(list) / chunks)
    return (list[i : i + n] for i in range(0, len(list), n))


LOCAL = threading.local()


def worker_process_batch(batch):
    """Grabs the worker's own processing function (with the correct "self")"""
    return LOCAL.process_batch(batch)


def input_generator(dataloader):
    """
    see https://github.com/pytorch/pytorch/issues/11201#issuecomment-486232056
    """
    for batch in dataloader:
        batch_copy = deepcopy(batch)
        del batch
        yield batch_copy


class Feature(metaclass=ABCMeta):
    """
    Features must have:
        - an process_batch function to take a batch of filenames and raw data and process it to a given embedding space
        - a list of Model objects needed to calculate the final embeddings
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "process_batch")
            and callable(subclass.initialize)
            #
            and hasattr(subclass, "models")
            and isinstance(
                subclass.models, List[Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction]]
            )
        )

    def __init__(self, dataset, batch_size, num_workers):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_generator = input_generator(
            DataLoader(
                dataset=dataset,
                batch_size=batch_size * num_workers,
                prefetch_factor=batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=self.collate,
                pin_memory=torch.cuda.is_available(),
            )
        )

        self.models = []

    def collate(self, batch):
        return [(fns, ims) for fns, ims in batch if ims is not None]

    def __getstate__(self):
        """Determine what parts of Feature object get pickled"""
        return {k: v for k, v in self.__dict__.items() if k != "input_generator"}  # generators can't be pickled

    def __setstate__(self, state):
        """Determine how to unpickle"""
        self.__dict__ = state

    def worker_init_fn(self):
        """Initialize models used for processing the feature on the correct device"""
        self = deepcopy(self)
        rank = threading.get_native_id()
        device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"

        for i, model in enumerate(self.models):
            model.initialize(device)

            if not isinstance(model.model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)):
                # TODO figure out a way to move this out of worker_init, this is executed on all workers --> inefficient
                # must be called AFTER model.initialize() though
                try:
                    # try to convert model to TorchScript which avoids python's GIL for heavy computation
                    self.models[i].model = torch.jit.script(model.model)
                except Exception as e:
                    print(e)
                    print(
                        f"WARNING: converting {self.models[i].__class__.__name__} to TorchScript failed, feature calculation might be slower..."
                    )

        # HACK to get around ThreadPool pickling of member functions. This results in the main thread's Feature object
        # being "self" rather than the worker's Feature object. Therefore we load the worker's process_batch function
        # into thread-local scope and execute it through the above helper function (worker_process_batch)
        LOCAL.process_batch = self.process_batch

    def loader(self):
        """A generator that processes batches in parallel"""
        with mp.pool.ThreadPool(self.num_workers, initializer=self.worker_init_fn) as pool:
            for batch in self.input_generator:
                for fns, embds in pool.imap_unordered(worker_process_batch, chunk(batch, self.batch_size)):
                    yield fns, embds

    def process(self):
        """Returns the processed features"""
        with tqdm(total=self.length) as progress:
            filenames, embeddings = [], []
            for fns, embds in self.loader():
                filenames += fns
                embeddings += embds
                progress.update(len(fns))
        return np.array(filenames), np.concatenate(embeddings)

    @abstractmethod
    def process_batch(self, batch: List[Tuple[str, np.ndarray]]) -> Tuple[List[str], List[np.ndarray]]:
        """Processes a batch of input data into the feature"""
        raise NotImplementedError
