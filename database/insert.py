from glob import glob

import features.registry
import torch

from database import Database

from .arguments import parse_insert_args


def insert(db_dir, img_dir, columns, batch_size=8, num_workers=2):
    db = Database(db_dir)

    files = glob(img_dir + "/*.jpeg") + glob(img_dir + "/*.jpg") + glob(img_dir + "/*.png")

    for feature in features.registry.retrieve(columns):
        print(f"Processing {feature.name}...")
        db.index(filenames=files, feature=feature.insert_fn(files, batch_size, num_workers), column_name=feature.name)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method("spawn")
    insert(**vars(parse_insert_args()))
