import os
from glob import glob
from time import time

import faiss
import joblib
import numpy as np
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

        print(f"Loading database with {len(self.id_file_map)} rows and {len(self.indices.keys())} columns")

    def index(self, filenames, feature, column_name=None, index_type="IDMap,Flat"):
        """Add new files to index"""
        t = time()
        if column_name is None:
            column_name = feature.__class__.__name__

        size = feature.size
        if not column_name in self.indices:
            self.indices[column_name] = faiss.index_factory(size, index_type)
        index = self.indices[column_name]

        files, features = feature.process()

        if not index.is_trained:
            index.train(index, features)

        ids = []
        for file in files:
            id = self.id_file_map.inverse.get(file, self.next_id)
            self.id_file_map[id] = file
            ids.append(id)
            self.next_id += 1
        ids = np.array(ids)

        index.add_with_ids(features.astype(np.float32), ids)

        faiss.write_index(index, f"{self.directory}/{column_name}.index")
        joblib.dump(self.id_file_map, self.map_file, compress=9)

        print(f"Finished adding {len(filenames)} files to column {column_name}")
        print(f"Took {time() - t} seconds")

    def random_sample(self, index, num_samples=1000, verbose=True):
        sample_ids = np.random.permutation(index.ntotal)[:num_samples]

        sample = []
        for sid in sample_ids:
            sample.append(index.reconstruct(int(sid)))
        sample = np.concatenate(sample)

        if verbose:
            print(faiss.MatrixStats(sample).comments)

        return sample

    def train_representative(self, index, num_samples=1000):
        index.train(self.random_sample(index, num_samples, verbose=False))

    def search(self, queries, columns, k=5, reduce=np.mean):
        """Search for queries in columns"""
        if not isinstance(queries, list):
            queries = [queries]
        results = {}
        for query in queries:
            for column_name in columns:
                colres = self.indices[column_name].search(query, k=2 * k)
                for dist, id in zip(colres[0].squeeze(), colres[1].squeeze()):
                    if id not in results:
                        results[id] = []
                    results[id].append(dist)
        return list(set([self.id_file_map[k] for k, _ in sorted(results.items(), key=lambda res: reduce(res[1]))]))[:k]
