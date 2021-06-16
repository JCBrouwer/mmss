from features.feature import Feature


class ProcessorFeature(Feature):
    def __init__(self, processor, dataset, batch_size, num_workers):
        super().__init__(dataset, batch_size, num_workers)
        self.processor = processor
        self.processors = [self.processor]
        self.size = processor.output_size

    def process_batch(self, batch):
        filenames, embeddings = [], []
        for filename, image in batch:
            filenames.append(filename)
            embeddings += self.processor(image)
        return filenames, embeddings


class ProcessorPipelineFeature(Feature):
    def __init__(self, processors, dataset, batch_size, num_workers):
        super().__init__(dataset, batch_size, num_workers)
        self.processors = processors
        self.size = processors[-1].output_size

    def process_batch(self, batch):
        filenames, embeddings = [], []
        for filename, image in batch:
            filenames.append(filename)

            output = image
            for processor in self.processors:
                output = processor(output)

            embeddings += output
        return filenames, embeddings
