import os

from util import download, extract_tgz

if not os.path.exists("cache/iaprtc12/images"):
    print("Downloading IAPR TC-12 dataset...")
    os.makedirs("cache/iaprtc12", exist_ok=True)
    download("http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz", "cache/iaprtc12/iaprtc12.tgz")
    extract_tgz("cache/iaprtc12/iaprtc12.tgz", "cache/iaprtc12/")
    os.remove("cache/iaprtc12/iaprtc12.tgz")
