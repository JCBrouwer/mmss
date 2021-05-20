import itertools
from collections import Counter, namedtuple
from dataclasses import dataclass
from typing import Callable, List

import features.registry
import matplotlib.pyplot as plt
import torch
from PIL import Image

from database import Database
from database.arguments import parse_search_args

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


@dataclass
class SearchColumns:
    search_fn: Callable
    column_names: List[str]


# cache the database and search models so we don't have to reload when searching multiple times
CachedValue = namedtuple("CachedValue", ("key", "val"))
DB = CachedValue("", None)
SEARCH_DATA = CachedValue("", None)


def search(db_dir, columns, num_results, query):
    global DB
    if DB.key != db_dir:
        DB = CachedValue(db_dir, Database(db_dir))

    global SEARCH_DATA
    if SEARCH_DATA.key != columns:
        feats = features.registry.retrieve(columns)

        # find the search model used for each feature
        search_fn_data_map = {}
        for feat in feats:
            if not type(feat.search_model) in search_fn_data_map:
                try:
                    rank = torch.multiprocessing.current_process()._identity[0]
                except:
                    rank = 0
                device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
                feat.search_model.initialize(device)
                search_fn_data_map[type(feat.search_model)] = SearchColumns(feat.search_model.search, [feat.name])
            else:
                search_fn_data_map[type(feat.search_model)].column_names.append(feat.name)
        SEARCH_DATA = CachedValue(columns, search_fn_data_map)

    # if no query supplied, solicit text input from user
    if query is None:
        query = input("Query: ")

    # get results from each search model
    filenames = []
    for search_data in SEARCH_DATA.val.values():
        query_embeddings = search_data.search_fn(query)
        fns, dists = DB.val.search(queries=query_embeddings, columns=search_data.column_names, k=num_results)
        filenames.append(fns)

    # populate front of list with results that are found in multiple columns, ordered by number of occurences
    counter = Counter(itertools.chain.from_iterable(filenames))
    results = [filename for filename, count in counter.most_common(num_results) if count > 1]

    # after that, interleave results from the front of each column (as long as its not already in the list)
    i = 0
    n = len(filenames)
    retries = 0
    while len(results) < num_results:
        try:
            filename = filenames[i % n][i // n]
            retries = 0
        except IndexError:
            retries += 1
            if retries == 10:
                break
            continue
        if filename not in results:
            results.append(filename)
        i += 1

    return results


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = parse_search_args()

    filenames_only = args.filenames_only
    del args.filenames_only

    results = search(**vars(args), query=None)

    if filenames_only:
        for file in results:
            print(file)
    else:
        fig, axarr = plt.subplots(2, len(results) // 2)
        for j in range(len(results)):
            axarr.flat[j].imshow(Image.open(results[j]))
            axarr.flat[j].axis("off")
        plt.tight_layout()
        plt.show()
