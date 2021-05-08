import features.registry
import matplotlib.pyplot as plt
import torch
from PIL import Image

from database import Database
from database.arguments import parse_search_args

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.set_grad_enabled(False)

    args = parse_search_args()

    db = Database(args.db_dir)

    feats = features.registry.retrieve(args.features)

    k = args.num_results

    # TODO how do we deal with cases where the features don't have the same search_fn??
    results = db.search(queries=feats[0].search_fn, columns=[feat.name for feat in feats], k=k)

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
