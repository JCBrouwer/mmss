# Multi-Media Similarity Search

```bash
# add images to search index
python -m database.insert /path/to/directory/of/images
python -m database.insert /path/to/another/directory
...
python -m database.insert /path/to/last/directory

# search database using text queries
python -m database.search
```
