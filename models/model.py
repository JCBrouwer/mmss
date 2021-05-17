from abc import ABCMeta, abstractmethod

import torch


class Model(metaclass=ABCMeta):
    """
    Models must have:
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
    def __call__(self, inputs):
        raise NotImplementedError
