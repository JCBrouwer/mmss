# Multi-Media Similarity Search

## Installation

To install this project we recommend using a [conda](https://docs.conda.io/en/latest/) environment with python version=3.8.

Clone the repository (and its submodules) and install the dependencies with the commands below.

```bash
git clone --recursive https://gitlab.ewi.tudelft.nl/cs4065/2020-2021/team23/multi-media-similarity-search.git
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install -r requirements.txt
```

The modules specified in requirements.txt do not always come together nicely, because the specifics of the requirements depend on the operating system you are using and your physical device configuration (whether or not you have a GPU that has CUDA compatible). If you run into errors with the dependencies see the [Troubleshooting section](#troubleshooting) below.

## Usage

```bash
python -m database.insert -h

# add images to search index
python -m database.insert /path/to/directory/of/images
python -m database.insert /path/to/another/directory
...
python -m database.insert /path/to/last/directory
```

```bash
python -m database.search -h

# search database using a text query
python -m database.search
```

## Architecture

```bash
cache/              # intermediate values and database files should be stored here
    modelzoo/       # pretrained model checkpoints etc.

database/
    db.py           # Main database implementation. Keeps track of different features in
                    # individual faiss indices. Provides functionality to search indices

    arguments.py    # Argument parsers for following scripts
    insert.py       # Script to insert new images into the database
    search.py       # Script to search the database based on text queries

processors/
    base.py         # The base Processor interface. This class encapsulates a specific feature processor
                    # or algorithm that transforms a single raw input data point into
                    # its corresponding embedding. Handles loading of feature processor weights and
                    # feature processor-specific preprocessing of raw data

    *.py            # Implementations of specific processors

features/
    data.py         # torch.Dataset implementations to load in raw images and videos

    feature.py      # The base Feature interface. This handles iterating over the data
                    # to transform it to the specified embedding space. Distributes work
                    # using multiprocessing / multi-GPU where applicable

    primitives.py   # Features that wrap around either a single Processor or multiple Processors.
                    # Generally different features can be implemented simply by writing
                    # the corresponding Processor class and wrapping its execution in one of
                    # these primitives

    registry.py     # Stores information on different features for easy access. This is
                    # where insertion and search procedures per feature are specified so
                    # that the rest of the code can rely on one unified API
```

## Troubleshooting
We describe the installation process if your configuration is not directly compatible with the requirements.txt.

_If you are working from a Linux distro and have a CUDA capable GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed and configured you can go directly to [step 3](#3-installing-via-requirementstxt) and everything should work_

### 1. Installing PyTorch

Since [CLIP](https://github.com/openai/CLIP) can cause some awkward dependency issues, installing this first seems to be the way to go without running  into issues later on.

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```
This installation assumes you have a CUDA capable device, replace the `cudatoolkit` version with the one installed on your device.
If your device does not have a GPU with CUDA capabilities, you can replace `cudatoolkit` with `cpuonly`.

### 2. Installing faiss
The next step should be installing [faiss](https://github.com/facebookresearch/faiss/blob/master). They have GPU and CPU variations of the package however the GPU variation is only available on Linux systems so if you are not installing from a Linux system then you have to install the CPU variation. Below are the three commands to execute depending on your system and specificity.

**CPU-only version:**
```bash
conda install -c pytorch faiss-cpu
```

**GPU without specific CUDA version:**
```bash
conda install -c pytorch faiss-gpu
```

**GPU for a specific CUDA version (the same version you specified in step 1)**
```bash
conda install -c pytorch faiss-gpu cudatoolkit=10.0
```

### 3. Installing via requirements.txt
Once these steps have been completed you should be able to install any of the missing packages directly. It is also
possible that you can do this without the first two steps given the right conditions, as explained in the installation overview.

```bash
pip install -r requirements.txt
```