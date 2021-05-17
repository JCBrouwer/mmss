import os
from glob import glob

import database
from util import download, extract_tgz


def load_list(fname):
    with open(fname, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_ap(positive, ambivalent, ranked_list):
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0

    intersect_size = 0
    j = 0
    for i in range(len(ranked_list)):
        if ranked_list[i] in ambivalent:
            continue
        if ranked_list[i] in positive:
            intersect_size += 1

        recall = intersect_size / len(positive)
        precision = intersect_size / (j + 1.0)

        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1

    return ap


def compute_ap(ranked_list, query):
    good = set(load_list(query.replace("query", "good")))
    ok = set(load_list(query.replace("query", "good")))
    junk = set(load_list(query.replace("query", "good")))

    return compute_ap(good | ok, junk, ranked_list)


if __name__ == "__main__":
    if not len(glob("cache/oxbuild/images/*")) > 0:
        os.makedirs("cache/oxbuild", exist_ok=True)
        os.makedirs("cache/oxbuild/images", exist_ok=True)
        download(
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz", "cache/oxbuild/oxbuild_images.tgz"
        )
        extract_tgz("cache/oxbuild/oxbuild_images.tgz", "cache/oxbuild/images/")
        os.remove("cache/oxbuild/oxbuild_images.tgz")

    if not len(glob("cache/oxbuild/annotations/*")) > 0:
        os.makedirs("cache/oxbuild/annotations", exist_ok=True)
        download(
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz", "cache/oxbuild/oxbuild_annot.tgz"
        )
        extract_tgz("cache/oxbuild/oxbuild_annot.tgz", "cache/oxbuild/annotations/")
        os.remove("cache/oxbuild/oxbuild_annot.tgz")

    if not os.path.exists("cache/oxbuild.db"):
        database.insert(db_dir="cache/oxbuild.db", img_dir="cache/oxbuild/images", columns="clip")

    for query in glob("cache/oxbuild/annotations/*query.txt"):
        print()

