from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor


class Processor(metaclass=ABCMeta):
    """
    Processors must have:
        - an __init__ function to initialize their object (this should only set information not load any large objects)
        - an initialize function to load the actual model weights into memory on the correct device
        - a __call__ function that takes a batch of inputs and transforms them to a batch of outputs
        - a model object which can be loaded to the correct device by PyTorch
        - an output_size denoting what size the embedding vector is
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "initialize")
            and callable(subclass.initialize)
            #
            and hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            #
            and hasattr(subclass, "model")
            and isinstance(subclass.model, (torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction))
            #
            and hasattr(subclass, "output_size")
            and isinstance(subclass.output_size, int)
        )

    @abstractmethod
    def initialize(self, device):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, inputs) -> List[np.ndarray]:
        raise NotImplementedError


class SearchProcessor(Processor, metaclass=ABCMeta):
    """
    Searchable model type. Not all processors have to be searchable hence the abstraction rather than adding an optional
    override to the base class Processor. Any Processor that can be searched in using any of the query types specified
    in the search descriptor should instead have this as their superclass.
    """

    @abstractmethod
    def search(self, query: List[Union[Image, str, Tensor]]):
        """
        Function to search this model for the specific query
        :param query: the query to be sought for, abstractly this can be any of the types defined in the Union
            but restrictions can be set by type hinting the implementation
        """
        raise NotImplementedError
