# Probe results

lancedb 0.41.0, pyarrow 26.0.1, Python 3.12.15 on Linux-7.0.0-1017-azure-x86_64-with-glibc2.43

| Finding | Status | Probe |
|---|---|---|
| F1 | Reproduced | Quickstart search step needs a package the install step leaves out |
| F2 | Reproduced | Vector index page teaches a create_index call the SDK deprecates |
| F3 | Reproduced | Mixing old and new create_index styles: what happens to index_type? |
| F4 | Reproduced | No-argument create_index() raises the legacy-API DeprecationWarning |
| F5 | Not reproduced | Too few rows to train an index: the docs give no minimum |
| F6 | Reproduced | create_index() in embedded LanceDB: blocking or asynchronous? |
| F7 | Reproduced | Vector type is inferred from the column name |
| F8 | Reproduced | Searching with a different metric than the index was built with |
| F9 | Reproduced | Scalar index page teaches create_scalar_index, which the SDK deprecates |

## Details

### F1: Quickstart search step needs a package the install step leaves out

- Docs say: Install with `pip install lancedb`; the Python search example ends in `.to_polars()`.
- Observed:

```json
{
  "to_polars": "ModuleNotFoundError: No module named 'polars'",
  "to_pandas": "ModuleNotFoundError: No module named 'pandas'"
}
```

### F2: Vector index page teaches a create_index call the SDK deprecates

- Docs say: table.create_index(metric='cosine', vector_column_name='keywords_embeddings', index_type='IVF_PQ')
- Observed:

```json
[
  "The create_index() API with metric/num_partitions parameters is deprecated and will be removed in a future version. Please migrate to the new unified API:"
]
```

### F3: Mixing old and new create_index styles: what happens to index_type?

- Docs say: index_type selects the index type (e.g. IVF_HNSW_SQ).
- Observed:

```json
{
  "indexes": [
    [
      "vector_idx",
      "IvfPq"
    ]
  ],
  "warnings": []
}
```

### F4: No-argument create_index() raises the legacy-API DeprecationWarning

- Docs say: OSS users can create vector indexes by calling table.create_index().
- Observed:

```json
[
  "The create_index() API with metric/num_partitions parameters is deprecated and will be removed in a future version. Please migrate to the new unified API:"
]
```

### F5: Too few rows to train an index: the docs give no minimum

- Docs say: No minimum row count or error is documented.
- Observed:

```json
{
  "10": "index built",
  "100": "index built",
  "255": "index built",
  "256": "index built"
}
```

### F6: create_index() in embedded LanceDB: blocking or asynchronous?

- Docs say: create_index() returns immediately, but the vector index builds asynchronously.
- Observed:

```json
{
  "listed_immediately": [
    "vector_idx"
  ],
  "num_unindexed_rows": 0
}
```

### F7: Vector type is inferred from the column name

- Docs say: Not mentioned on the quickstart, multimodal, vector index or scalar index pages.
- Observed:

```json
{
  "embedding": {
    "type": "fixed_size_list<item: float>[16]",
    "search": "search worked"
  },
  "features": {
    "type": "list<item: double>",
    "search": "ValueError: Invalid input, Data type is not a vector (FixedSizeListArray or List<FixedSizeListArray>), but Float64"
  }
}
```

### F8: Searching with a different metric than the index was built with

- Docs say: A different metric cannot use the index and falls back to a flat search.
- Observed:

```json
{
  "omitted": "uses index (ANN)",
  "cosine": "uses index (ANN)",
  "l2": "full scan (no ANN node)"
}
```

### F9: Scalar index page teaches create_scalar_index, which the SDK deprecates

- Docs say: tbl.create_scalar_index("publisher", index_type="BITMAP")
- Observed:

```json
[
  "create_scalar_index is deprecated as of 0.25.0. Use create_index() with config=BTree()/Bitmap()/LabelList() instead."
]
```
