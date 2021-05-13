import matplotlib.pyplot as plt
import torch
from PIL import Image

import features.registry
from database import Database
from database.arguments import parse_search_args

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    args = parse_search_args()

    db = Database(args.db_dir)

    feats = features.registry.retrieve(args.features)

    k = args.num_results

    search_functions = {}
    for feat in feats:
        if not type(feat.end_search) in search_functions:
            search_functions[type(feat.end_search)] = feat.end_search.search

    results = []
    for search_fn in search_functions.values():
        results.append(db.search(queries=search_fn, columns=[feat.name for feat in feats], k=k))

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
