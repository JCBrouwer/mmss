from models.clip import Clip

from features.base import Feature


class ClipFeature(Feature):
    def __init__(self, dataset, batch_size, num_workers, backbone="ViT-B/32"):
        super().__init__(dataset, batch_size, num_workers)
        self.clip = Clip(backbone)
        self.models = [self.clip]
        self.size = 512

    def process_batch(self, batch):
        """Processes a batch of input data into the feature"""
        filenames, embeddings = [], []
        for filename, image in batch:
            filenames.append(filename)
            embeddings.append(self.clip(image))
        return filenames, embeddings
