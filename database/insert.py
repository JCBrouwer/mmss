import sys
from glob import glob

import torch

from database import Database
from features.clip import ClipFeature
from dataloader import Images

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    db = Database("cache/db")

    directory = sys.argv[1]
    files = glob(directory + "/*.jpeg") + glob(directory + "/*.jpg") + glob(directory + "/*.png")

    db.index(files, ClipFeature(Images(files), batch_size=64, num_workers=4))
