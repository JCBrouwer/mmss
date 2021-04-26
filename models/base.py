from abc import abstractmethod


class Model:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, device):
        pass

    @abstractmethod
    def __call__(self, images):
        pass
