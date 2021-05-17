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


@dataclass
class SearchColumns:
    search_fn: Callable
    column_names: List[str]


CachedDB = namedtuple("CachedDB", ("dir", "db"))
DB = CachedDB("", None)


def search(db_dir, columns, num_results, query):
    global DB
    if DB.dir != db_dir:
        DB = CachedDB(db_dir, Database(db_dir))  # avoid reloading database when searching multiple times

    feats = features.registry.retrieve(columns)

    # find the search model used for each feature
    search_fn_data_map = {}
    for feat in feats:
        if not type(feat.search_model) in search_fn_data_map:
            feat.search_model.initialize("cpu")
            search_fn_data_map[type(feat.search_model)] = SearchColumns(feat.search_model.search, [feat.name])
        else:
            search_fn_data_map[type(feat.search_model)].column_names.append(feat.name)

    # if no query supplied, solicit text input from user
    if query is None:
        query = input("Query: ")

    # get results from each search model
    results = []
    for search_data in search_fn_data_map.values():
        query_embeddings = search_data.search_fn(query)
        results.append(DB.db.search(queries=query_embeddings, columns=search_data.column_names, k=num_results))

    # populate front of list with results that are found in multiple columns, ordered by number of occurences
    counter = Counter(list(itertools.chain.from_iterable(results)))
    final_results = [filename for filename, count in counter.most_common(num_results) if count > 1]

    # after that, interleave results from the front of each column (as long as its not already in the list)
    i = 0
    while len(final_results) < num_results:
        filename = results[i % len(results)][i // len(results)]
        if filename not in final_results:
            final_results.append(filename)
        i += 1

    return final_results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
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
