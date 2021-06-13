import os
import time
from glob import glob

from database import Database
from features import registry
from features.registry import RegistryEntry
from util import download, extract_tgz

OXFORD_CACHE = "cache/oxbuild/images"
ANNOTATION_CACHE = "cache/oxbuild/annotations"
OXFORD_IMAGE_DOWNLOAD = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
OXFORD_ANNOTATION_DOWNLOAD = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz"
DATABASE_DIR = "cache/oxbuild.db"


def download_cache(directory, url):
    if not len(glob(f"{directory}/*")) > 0:
        os.makedirs(directory, exist_ok=True)
        tgz_name = f'{directory}.tgz'
        download(url, tgz_name)
        extract_tgz(tgz_name, directory)
        os.remove(tgz_name)


def download_ni_cache():
    os.makedirs("cache/oxbuild", exist_ok=True)
    download_cache(OXFORD_CACHE, OXFORD_IMAGE_DOWNLOAD)
    download_cache(ANNOTATION_CACHE, OXFORD_ANNOTATION_DOWNLOAD)


def run_performance_analysis(reg_entry: RegistryEntry):
    print(f"Performance analysis over column: {reg_entry.name}")
    db = Database(DATABASE_DIR)

    # Single worker insertion
    files = glob(OXFORD_CACHE + "/*.jpg")

    seconds = time.time()

    # Single worker insertion with batch size of 8
    db.index(
        feature=reg_entry.insert_fn(files, 8, 1, reg_entry.search_model), column_name=reg_entry.name,
    )

    index_time = round(time.time() - seconds, 1)
    print(f"Seconds to index {reg_entry.name} for {len(files)} images = {index_time} seconds")

    return index_time, round(index_time / len(files), 2)


# Query length performance
# Indexing performance analysis
# Feature performance analysis
# Google Colab Performance

if __name__ == '__main__':
    print("Analyzing performance on Oxford Buildings")
    download_ni_cache()

    column_options = registry.retrieve(['all'])
    index_performance_dict = dict()
    for entry in column_options:
        total_time, time_per_image = run_performance_analysis(entry)
        index_performance_dict[entry.name] = (total_time, time_per_image)
