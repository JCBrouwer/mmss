import copy
import json
import os
import zipfile

import database
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from util import download

img_dir = "cache/coco/val2014/"
db_dir = "cache/coco.db"
columns = ["clip"]
n_workers = mp.cpu_count()

if not os.path.exists("cache/coco/dataset_coco.json"):
    path, _ = download("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip")
    with zipfile.ZipFile(path, "r") as f:
        f.extractall("cache/coco/")
    os.remove(path)
    os.remove("cache/coco/dataset_flickr30k.json")
    os.remove("cache/coco/dataset_flickr8k.json")

    # filter out only validation set images to keep things manageable
    with open("cache/coco/dataset_coco.json", "r") as f:
        info = json.load(f)
    annotations = copy.deepcopy(info["images"])
    del info
    filtered = []
    for annot in annotations:
        if "train" not in annot["filename"]:
            filtered.append({"filename": annot["filename"], "sentences": annot["sentences"]})
    with open("cache/coco/karpathy_val.json", "w") as f:
        json.dump(filtered, f)

if not os.path.exists(img_dir):
    os.makedirs("cache/coco/", exist_ok=True)
    path, _ = download("http://images.cocodataset.org/zips/val2014.zip")
    with zipfile.ZipFile(path, "r") as f:
        f.extractall("cache/coco/")
    os.remove(path)

if not os.path.exists(db_dir):
    database.insert(db_dir=db_dir, img_dir=img_dir, columns=columns)

with open("cache/coco/karpathy_val.json", "r") as f:
    annotations = json.load(f)


def retrieve(index):
    caption = annotations[index // 5]["sentences"][index % 5]["raw"]
    results = database.search(db_dir, columns=columns, num_results=25, query=caption)
    image_file = img_dir + annotations[index // 5]["filename"]
    try:
        return results.index(image_file)
    except ValueError:
        return len(results)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    ranks = []
    with mp.Pool(n_workers) as pool:
        indices = np.random.permutation(5 * len(annotations))
        for rank in tqdm(pool.imap_unordered(retrieve, indices), total=len(indices)):
            ranks.append(rank)

    rank = [1, 5, 10]
    acc = [sum([_ < r for _ in ranks]) / len(ranks) for r in rank]
    print(f"Retrieval: {acc[0]:.4f} @ R1, {acc[1]:.4f} @ R5, {acc[2]:.4f} @ R10")
