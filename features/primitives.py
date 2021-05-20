import torch

from features.feature import Feature


class ModelFeature(Feature):
    def __init__(self, model, dataset, batch_size, num_workers):
        super().__init__(dataset, batch_size, num_workers)
        self.model = model
        self.models = [self.model]
        self.size = model.output_size

    def process_batch(self, batch):
        filenames, embeddings = [], []
        for filename, image in batch:
            filenames.append(filename)
            embeddings += self.model(image)
        return filenames, embeddings


def transpose(list_of_lists):
    return tuple(map(list, zip(*list_of_lists)))


class ModelPipelineFeature(Feature):
    def __init__(self, models, dataset, batch_size, num_workers):
        super().__init__(dataset, batch_size, num_workers)
        self.models = models
        self.size = models[-1].output_size

    def process_batch(self, batch):
        filenames, embeddings = [], []
        for filename, image in batch:
            filenames.append(filename)

            output = image
            for model in self.models:
                output = model(output)

            embeddings += output
        return filenames, embeddings
