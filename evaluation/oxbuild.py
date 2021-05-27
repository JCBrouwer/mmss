import os
from glob import glob
from pathlib import Path

import database
import numpy as np
from PIL import Image
from util import download, extract_tgz
from tqdm import tqdm


def compute_ap(ranked_list, query):
    def load_list(fname):
        with open(fname, "r") as f:
            return [line.strip() for line in f.readlines()]

    def calculate(positive, ambivalent, ranked_list):
        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0

        intersect_size = 0
        j = 0
        for image in ranked_list:
            image_stem = Path(image).stem
            if image_stem in ambivalent:
                continue
            if image_stem in positive:
                intersect_size += 1

            recall = intersect_size / len(positive)
            precision = intersect_size / (j + 1.0)

            ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

            old_recall = recall
            old_precision = precision
            j += 1

        return ap

    good = set(load_list(query.replace("query", "good")))
    ok = set(load_list(query.replace("query", "ok")))
    junk = set(load_list(query.replace("query", "junk")))
    return calculate(good | ok, junk, ranked_list)


if __name__ == "__main__":
    print("Running evaluation on Oxford Buildings...")

    img_dir = "cache/oxbuild/images"
    annot_dir = "cache/oxbuild/annotations"

    if not len(glob(f"{img_dir}/*")) > 0:
        os.makedirs("cache/oxbuild", exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        download("https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz", f"{img_dir}.tgz")
        extract_tgz(f"{img_dir}.tgz", img_dir)
        os.remove(f"{img_dir}.tgz")

    if not len(glob(f"{annot_dir}/*")) > 0:
        os.makedirs(f"{annot_dir}", exist_ok=True)
        download("https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz", f"{annot_dir}.tgz")
        extract_tgz(f"{annot_dir}.tgz", f"{annot_dir}/")
        os.remove(f"{annot_dir}.tgz")

    db_dir = f"cache/oxbuild.db"

    for column_set in [
        ["clip"],
        ["sift"],
        ["orb"],
        ["brisk"],
        ["clip", "sift"],
        ["clip", "orb"],
        ["clip", "brisk"],
        ["sift", "orb", "brisk"],
        ["clip", "sift", "orb", "brisk"],
    ]:
        print(column_set)
        indices_exist = all(any(col in index for index in glob(db_dir + "/*.index")) for col in column_set)
        if not indices_exist:
            database.insert(db_dir=db_dir, img_dir=img_dir, columns=column_set, num_workers=8)

        aps = []
        for query in tqdm(sorted(glob(f"{annot_dir}/*query.txt"))):
            with open(query, "r") as f:
                file_str = f.readlines()[0].strip().split(" ")[0]
            img_file = img_dir + "/" + file_str.replace("oxc1_", "") + ".jpg"

            results = database.search(db_dir, column_set, num_results=25, query=Image.open(img_file))
            ap = compute_ap(results, query)
            aps.append(ap)

        print(
            f"average precision: \t\t min = {np.min(aps):.3f} \t\t median = {np.median(aps):.3f} \t\t mean = {np.mean(aps):.3f} \t\t max = {np.max(aps):.3f}"
        )
