import matplotlib.pyplot as plt
import numpy as np
import torch
from models.clip import Clip
from PIL import Image

from database import Database

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    db = Database("cache/db")
    clip = Clip()
    clip.initialize("cuda")

    k = 8
    for _ in range(5):
        fig, axarr = plt.subplots(2, k // 2)
        cluster = db.search(queries=np.array(clip(input("Query: ")), dtype=np.float32), columns=["ClipFeature"], k=k)

        for j in range(k):
            axarr.flat[j].imshow(Image.open(cluster[j]))
            axarr.flat[j].axis("off")
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)
    plt.show()
