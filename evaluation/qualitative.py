"""
Pre-calculated features from:
https://colab.research.google.com/github/haltakov/natural-language-image-search/blob/main/colab/unsplash-image-search.ipynb
"""

import os
from pathlib import Path
from time import time

import faiss
import joblib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from database import Database
from database.search import jegou_criterion
from PIL import Image
from processors import Clip


def display(ax, url):
    ax.imshow(Image.open(requests.get(f"{url}/download?w=320", stream=True).raw))
    pc = mpl.collections.PathCollection([mpl.text.TextPath((2, 4), "Photo from Unsplash", size=0.1)])
    pc.set_urls([url])
    ax.add_collection(pc)
    plt.draw()


def slerp(val, low, high):
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def geodesic_mean(embeddings):
    if len(embeddings) == 1:
        return embeddings
    elif len(embeddings) == 2:
        return [slerp(0.5, embeddings[0], embeddings[1])[None, :]]
    else:
        return np.mean(embeddings, axis=0, keepdims=True)


if __name__ == "__main__":
    data_dir = "cache/unsplash-dataset/"
    db_dir = f"cache/qualitative.db"
    index_type = "IDMap,Flat"
    num_results = 4

    db = Database(db_dir)

    if not os.path.exists(f"{db_dir}/clip-image-embedding.index"):
        t = time()

        photo_ids = pd.read_csv(f"{data_dir}/photo_ids.csv")["photo_id"]
        features = np.load(f"{data_dir}/features.npy").astype(np.float32)
        print(features.shape)
        size = features.shape[-1]

        db.indices["clip-image-embedding"] = faiss.index_factory(size, index_type)
        index = db.indices["clip-image-embedding"]
        if faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), db.rank, index)

        if not index.is_trained:
            index.train(features)

        ids = []
        for i, id in enumerate(photo_ids):
            db.id_file_map[i] = f"https://unsplash.com/photos/{id}"
            ids.append(i)
        ids = np.array(ids)

        index.add_with_ids(features, ids)

        if faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, f"{db.directory}/clip-image-embedding.index")
        joblib.dump(db.id_file_map, db.map_file)

        print(f"Finished adding {len(ids)} entries to column clip-image-embedding")
        print(f"Took {time() - t} seconds")

    clip = Clip()
    clip.initialize("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(5):
        out_file = []
        query = input("Query: ")
        query = query.split("|")
        reference = None
        for i, q in enumerate(query):
            q = q.strip()
            if "/" in q:
                reference = Image.open(q)
                query[i] = reference
                out_file += [Path(q).stem]
            else:
                out_file += [q]
        out_file = " ".join(out_file)

        filenames, distances = db.search(
            queries=geodesic_mean(clip.search(query)), columns=["clip-image-embedding"], k=num_results
        )
        results = jegou_criterion([filenames], [distances], num_results)

        fig, axarr = plt.subplots(1, len(results) + (1 if reference else 0), figsize=((5 if reference else 4) * 4, 4))
        for j in range(len(results)):
            display(axarr.flat[j], results[j])
            axarr.flat[j].axis("off")
        if reference:
            axarr.flat[-1].imshow(reference)
            axarr.flat[-1].set_title("Reference")
            axarr.flat[-1].axis("off")
        plt.tight_layout()
        plt.savefig(f"cache/{out_file}.png")
