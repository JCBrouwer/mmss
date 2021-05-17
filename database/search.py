import features.registry
import matplotlib.pyplot as plt
import torch
from PIL import Image

from database import Database
from database.arguments import parse_search_args


def search(db_dir, columns, num_results):
    db = Database(db_dir)

    feats = features.registry.retrieve(columns)

    # TODO how do we deal with cases where the features don't have the same search_fn??
    return db.search(queries=feats[0].search_fn, columns=[feat.name for feat in feats], k=num_results)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method("spawn")

    args = parse_search_args()

    filenames_only = args.filenames_only
    del args.filenames_only

    results = search(**vars(args))

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
