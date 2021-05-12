from glob import glob

import features.registry
import torch

from database import Database

from .arguments import parse_insert_args

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    args = parse_insert_args()

    db = Database(args.db_dir)

    files = glob(args.img_dir + "/*.jpeg") + glob(args.img_dir + "/*.jpg") + glob(args.img_dir + "/*.png")

    for feature in features.registry.retrieve(args.features):
        print(f"Processing {feature.name}...")
        db.index(
            filenames=files,
            feature=feature.insert_fn(files, args.batch_size, args.num_workers),
            column_name=feature.name,
        )
