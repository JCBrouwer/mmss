from dataclasses import dataclass
from typing import Callable, List

import matplotlib.pyplot as plt
import torch
from PIL import Image

import features.registry
from database import Database
from database.arguments import parse_search_args


def search(db_dir, columns, num_results, query):
    db = Database(db_dir)

    feats = features.registry.retrieve(columns)

    search_fn_data_map = {}
    for feat in feats:
        if not type(feat.search_model) in search_fn_data_map:
            feat.search_model.initialize("cpu")
            search_fn_data_map[type(feat.search_model)] = SearchColumns(feat.search_model.search, [feat.name])
        else:
            search_fn_data_map[type(feat.search_model)].column_names.append(feat.name)

    if query is None:
        query = input("Query: ")

    results = []
    for search_data in search_fn_data_map.values():
        query_embeddings = search_data.search_fn(query)
        results.extend(db.search(queries=query_embeddings, columns=search_data.column_names, k=num_results))

    return results


@dataclass
class SearchColumns:
    search_fn: Callable
    column_names: List[str]


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
