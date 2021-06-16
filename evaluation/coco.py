import copy
import json
import os
import zipfile

import database
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from util import download
from glob import glob

img_dir = "cache/coco/val2014/"
db_dir = "cache/coco.db"
columns = ["clip", "yolo"]
n_workers = 8

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

indices_exist = [any(col in index for index in glob(db_dir + "/*.index")) for col in columns]
if not all(indices_exist):
    database.insert(
        db_dir=db_dir,
        img_dir=img_dir,
        columns=[col for col, exist in zip(columns, indices_exist) if not exist],
        num_workers=2,
    )

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
        indices = np.random.permutation(5 * len(annotations))[: len(annotations) // 2]
        for rank in tqdm(pool.imap_unordered(retrieve, indices), total=len(indices)):
            ranks.append(rank)

    rank = [1, 5, 10]
    acc = [sum([_ < r for _ in ranks]) / len(ranks) for r in rank]
    print(f"Retrieval: {acc[0]:.4f} @ R1, {acc[1]:.4f} @ R5, {acc[2]:.4f} @ R10")
