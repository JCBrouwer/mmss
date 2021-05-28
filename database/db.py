import os
from glob import glob
from time import time

import faiss
import joblib
import numpy as np
import torch
from bidict import bidict


class Database:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        # bidirectional map of faiss index IDs to filenames
        self.map_file = self.directory + "/id_file.pkl"
        try:
            self.id_file_map = joblib.load(self.map_file)
            self.next_id = np.max(list(self.id_file_map.keys())) + 1
        except:
            self.id_file_map = bidict()
            self.next_id = 0

        # each column in the database corresponds to a faiss index over those embeddings
        self.indices = {}
        for index_file in glob(self.directory + "/*.index"):
            column_name = os.path.basename(index_file).replace(".index", "")
            self.indices[column_name] = faiss.read_index(index_file)

        print(f"Loading database with {len(self.id_file_map)} files and {len(self.indices.keys())} columns")

        try:
            self.rank = torch.multiprocessing.current_process()._identity[0] % torch.cuda.device_count()
        except:
            self.rank = 0

    def index(self, feature, column_name=None, index_type="IDMap,Flat"):
        """Add new files to index"""
        t = time()
        if column_name is None:
            column_name = feature.__class__.__name__

        # process the feature
        files, features = feature.process()

        # if feature doesn't return single vectors (e.g. SIFT), flatten to longer list of single vectors
        if len(features[0].shape) > 1:
            expanded_files = []
            for file, feat in zip(files, features):
                expanded_files += [file] * len(feat)
            files = np.array(expanded_files)
            features = np.concatenate(features, axis=0)
        features = features.astype(np.float32)

        # check if index for this feature already exists, otherwise create it
        size = feature.size
        if not column_name in self.indices:
            self.indices[column_name] = faiss.index_factory(size, index_type)
        index = self.indices[column_name]
        if faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.rank, index)

        # if index needs training, train
        if not index.is_trained:
            index.train(features)

        # get ids for each file, some might already be present in our id_file_map
        ids = []
        for file in files:
            id = self.id_file_map.inverse.get(file, self.next_id)
            self.id_file_map[id] = file
            ids.append(id)
            self.next_id += 1
        ids = np.array(ids)

        # insert to the index and write everything to disk
        index.add_with_ids(features, ids)

        if faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, f"{self.directory}/{column_name}.index")
        joblib.dump(self.id_file_map, self.map_file, compress=9)

        print(f"Finished adding {len(files)} entries to column {column_name}")
        print(f"Took {time() - t} seconds")

    def random_sample(self, index, num_samples, verbose=True):
        sample_ids = np.random.permutation(index.ntotal)[:num_samples]

        sample = []
        for sid in sample_ids:
            sample.append(index.reconstruct(int(sid)))
        sample = np.concatenate(sample)

        if verbose:
            print(faiss.MatrixStats(sample).comments)

        return sample

    def train_representative(self, index, num_samples=10_000):
        index.train(self.random_sample(index, num_samples, verbose=False))

    def upgrade_indices(self, new_index_type="IDMap,IVF100,PQ8"):
        for column_name, index in self.indices.items():
            if faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.rank, index)

            vectors = index.reconstruct_n(0, index.ntotal)
            ids = np.array([index.id_map.at(i) for i in range(index.id_map.size())])
            assert len(vectors) == len(ids)

            new_index = faiss.index_factory(vectors.shape[1], new_index_type)
            if faiss.get_num_gpus() > 0:
                new_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.rank, new_index)

            if not new_index.is_trained:
                new_index.train(vectors)

            new_index.add_with_ids(vectors, ids)

            if faiss.get_num_gpus() > 0:
                new_index = faiss.index_gpu_to_cpu(new_index)

            faiss.write_index(new_index, f"{self.directory}_new/{column_name}.index")

    def search(self, queries, columns, k=25, reduce=np.sum, jegou_criterion=True):
        """Search for queries in columns"""
        if not isinstance(queries, list):
            queries = [queries]
        results = {}
        for column_name in columns:

            index = self.indices[column_name]
            if faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.rank, index)

            for query in queries:
                distances, ids = index.search(query, k=k)
                print(ids)
                if distances.shape[0] == 1:
                    for dist, id in zip(distances.squeeze(), ids.squeeze()):
                        if id not in results:
                            results[id] = []
                        results[id].append(dist)
                else:
                    for x in range(len(distances)):
                        for y in range(k):

                            if jegou_criterion:
                                distances[x, y] = -max(distances[x, -1] - distances[x, y], 0)

                            if ids[x, y] not in results:
                                results[ids[x, y]] = []
                            results[ids[x, y]].append(distances[x, y])

        # sort results by number of occurrences of file_id, break ties by distance (-length to sort descending)
        best_results = sorted(results.items(), key=lambda id_dists: reduce(id_dists[1]))[:k]

        filenames, distances = [], []
        for id, dist in best_results:
            if id == -1:
                continue
            print(id, dist)
            filenames.append(self.id_file_map[id])
            distances.append(reduce(dist))

        return filenames, distances
