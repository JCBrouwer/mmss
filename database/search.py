from dataclasses import dataclass
from typing import Callable, List

import matplotlib.pyplot as plt
import torch
from PIL import Image

import features.registry
from database import Database
from database.arguments import parse_search_args

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


@dataclass
class SearchColumns:
    search_fn: Callable
    column_names: List[str]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    args = parse_search_args()

    db = Database(args.db_dir)

    feats = features.registry.retrieve(args.features)

    k = args.num_results

    search_fn_data_map = {}
    for feat in feats:
        if not type(feat.search_model) in search_fn_data_map:
            search_fn_data_map[type(feat.search_model)] = SearchColumns(feat.search_model.search, [feat.name])
        else:
            search_fn_data_map[type(feat.search_model)].column_names.append(feat.name)

    results = []
    for search_data in search_fn_data_map.values():
        results.append(db.search(queries=search_data.search_fn, columns=search_data.column_names, k=k))

    if args.filenames_only:
        for file in results:
            print(file)
    else:
        fig, axarr = plt.subplots(2, k // 2)
        for j in range(k):
            axarr.flat[j].imshow(Image.open(results[j]))
            axarr.flat[j].axis("off")
        plt.tight_layout()
        plt.show()
