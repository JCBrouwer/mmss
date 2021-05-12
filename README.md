# Multi-Media Similarity Search

## Installation / Usage

```bash
pip install -r requirements.txt
```

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

```
cache/              # intermediate values and database files should be stored here
    modelzoo/       # pretrained model checkpoints etc.

database/
    db.py           # Main database implementation. Keeps track of different features in
                    # individual faiss indices. Provides functionality to search indices

    arguments.py    # Argument parsers for following scripts
    insert.py       # Script to insert new images into the database
    search.py       # Script to search the database based on text queries

models/
    model.py        # The base Model interface. This class encapsulates a specific model
                    # or algorithm that transforms a single raw input data point into
                    # its corresponding embedding. Handles loading of model weights and
                    # model-specific preprocessing of raw data

    *.py            # Implementations of specific models

features/
    data.py         # torch.Dataset implementations to load in raw images and videos

    feature.py      # The base Feature interface. This handles iterating over the data
                    # to transform it to the specified embedding space. Distributes work
                    # using multiprocessing / multi-GPU where applicable

    primitives.py   # Features that wrap around either a single Model or multiple Models.
                    # Generally different features can be implemented simply by writing
                    # the corresponding Model class and wrapping its execution in one of
                    # these primitives

    registry.py     # Stores information on different features for easy access. This is
                    # where insertion and search procedures per feature are specified so
                    # that the rest of the code can rely on one unified API
```