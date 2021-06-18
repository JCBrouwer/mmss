import gc
import os
import shutil
import time
from glob import glob

import database
import psutil
import torch.cuda
from database import Database
from features import registry
from features.registry import RegistryEntry
from PIL import Image
from util import download, extract_tgz

OXFORD_CACHE = "cache/oxbuild/images"
ANNOTATION_CACHE = "cache/oxbuild/annotations"
OXFORD_IMAGE_DOWNLOAD = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
OXFORD_ANNOTATION_DOWNLOAD = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz"
DATABASE_DIR = "cache/oxbuild.db"


N_WORKERS = 4


def download_cache(directory, url):
    if not len(glob(f"{directory}/*")) > 0:
        os.makedirs(directory, exist_ok=True)
        tgz_name = f"{directory}.tgz"
        download(url, tgz_name)
        extract_tgz(tgz_name, directory)
        os.remove(tgz_name)


def download_ni_cache():
    os.makedirs("cache/oxbuild", exist_ok=True)
    download_cache(OXFORD_CACHE, OXFORD_IMAGE_DOWNLOAD)
    download_cache(ANNOTATION_CACHE, OXFORD_ANNOTATION_DOWNLOAD)


def remove_db_if_exists():
    if os.path.exists(DATABASE_DIR):
        # Remove dir incl contents
        shutil.rmtree(DATABASE_DIR)


def run_performance_analysis(reg_entry: RegistryEntry):
    print(f"Performance analysis over column: {reg_entry.name}")
    db = Database(DATABASE_DIR)

    files = glob(OXFORD_CACHE + "/*.jpg")[:1000]

    seconds = time.time()

    # batch size of 8
    processor = reg_entry.insert_fn(files, 8, N_WORKERS, reg_entry.search_model)
    db.index(feature=processor, column_name=reg_entry.name)
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    index_time = time.time() - seconds
    print(f"Seconds to index {reg_entry.name} for {len(files)} images = {index_time} seconds")

    return index_time, index_time / len(files)


def run_query_performance_analysis(columns):
    query_times = []
    for query_file in sorted(glob(f"{ANNOTATION_CACHE}/*query.txt")):
        with open(query_file, "r") as f:
            query_str = f.readlines()[0].strip().split(" ")[0]
        formatted_query = query_str.replace("oxc1_", "") + ".jpg"
        img_file = os.path.join(OXFORD_CACHE, formatted_query)
        start = time.time()
        database.search(DATABASE_DIR, columns, num_results=10, query=Image.open(img_file))
        taken = time.time() - start
        query_times.append(taken)
    return query_times


def get_size(byte_amount):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if byte_amount < factor:
            return f"{byte_amount:.2f}{unit}B"
        byte_amount /= factor


def print_system_specs():
    svmem = psutil.virtual_memory()
    cores = psutil.cpu_count(logical=False)
    vcores = psutil.cpu_count(logical=True)
    cpufreq = psutil.cpu_freq()

    print("#" * 40, "System Overview", "#" * 40)
    print(f"CPU: {vcores} cores ({vcores - cores} virtual) - {cpufreq.max:.2f}Mhz")
    print(f"RAM: {get_size(svmem.total)}")
    if torch.cuda.is_available():
        for gpu_index in range(torch.cuda.device_count()):
            print(f"GPU_{gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    else:
        print("GPU: Not available for PyTorch")


def print_indexing_performance():
    total_indexing_time = sum([x[0] for x in index_performance_dict.values()])
    total_time_per_image = sum(x[1] for x in index_performance_dict.values())
    avg_indexing_time = total_indexing_time / len(index_performance_dict.keys())
    avg_indexing_time_image = total_indexing_time / len(index_performance_dict.keys())

    print("#" * 40, "Feature Pipeline Performance", "#" * 40)
    print(f"Total time: {total_indexing_time}")
    print(f"Total time per image: {total_time_per_image}")
    print(f"Average time over all features: {avg_indexing_time}")
    print(f"Average time per image over all features: {avg_indexing_time_image}")
    for key in index_performance_dict.keys():
        analysis = index_performance_dict[key]
        print()
        print(f"  {key}:")
        print(f"  Time: {analysis[0]}")
        print(f"  Time per image: {analysis[1]} ")


def print_query_performance():
    all_query_performance = query_performance_dict["all"]
    total_time_all = sum(all_query_performance)
    average_query_time = total_time_all / len(all_query_performance)
    print("#" * 40, "Query Performance", "#" * 40)
    print(f"Total time: {total_time_all}")
    print(f"Average time per query: {average_query_time}")

    for key in query_performance_dict.keys():
        if key == "all":
            continue
        total_time_key = sum(query_performance_dict[key])
        average_time_key = total_time_key / len(query_performance_dict[key])
        print()
        print(f"  {key}:")
        print(f"  Total time for queries: {total_time_key}")
        print(f"  Average time per query: {average_time_key}")


if __name__ == "__main__":
    print("Analyzing performance on Oxford Buildings")
    download_ni_cache()
    remove_db_if_exists()

    index_performance_dict = dict()
    for entry in registry.REGISTRY:
        total_time, time_per_image = run_performance_analysis(registry.REGISTRY[entry])
        index_performance_dict[entry] = (total_time, time_per_image)

    query_performance_dict = dict()
    for entry in registry.REGISTRY:
        query_performance_dict[entry] = run_query_performance_analysis([entry])
    query_performance_dict["all"] = run_query_performance_analysis(["all"])

    print_system_specs()
    print("Number of workers: ", N_WORKERS)
    print_indexing_performance()
    print_query_performance()
