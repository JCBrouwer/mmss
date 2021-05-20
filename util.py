import tarfile
from urllib.request import urlretrieve

from tqdm import tqdm


def show_progress(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download(url, path):
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading to: "+path) as t:
        return urlretrieve(url, path, reporthook=show_progress(t))


def extract_tgz(filepath, extract_path):
    tar = tarfile.open(filepath, "r")
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tgz(item.name, "./" + item.name[: item.name.rfind("/")])
