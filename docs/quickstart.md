# Find similar images with LanceDB

Store images, their embeddings and their metadata in one LanceDB table, then find
the images that look most like a query image, filter by metadata, and add indexes.

You'll build an image search over 1,797 small handwritten-digit pictures. Search
for images similar to a handwritten 0 and you get back other handwritten 0s.

**Time:** about 10 minutes. **Runs:** fully offline after you install packages.
No account, API key or GPU needed.

## Before you start

| Requirement | Details |
|---|---|
| Python | 3.10 to 3.13 |
| OS | Linux, macOS or Windows on x86_64 or ARM64. On x86_64 CPUs older than Intel Haswell (2013), see [Troubleshooting](#troubleshooting). |
| Disk | About 5 MB for the database |
| Packages | `lancedb` (tested with 0.41.0), `pillow`, `scikit-learn`. LanceDB installs `pyarrow` and `numpy` for you. |

Install the packages:

```bash
pip install "lancedb==0.41.0" pillow scikit-learn
```

You don't need pandas or polars: this guide reads results with `.to_list()`,
which returns plain Python dicts. If you prefer DataFrames, see
[Get results as a DataFrame](#get-results-as-a-dataframe).

Run every code block below in order, in one Python session or one script. Each
block depends on the variables the earlier blocks define.

## 1. Prepare images and embeddings

Load the digits dataset that ships with scikit-learn, so nothing is downloaded.
For each image, build one record: the picture as PNG bytes, an embedding vector,
and metadata to filter on.

```python
import io

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()  # 1,797 images, 8x8 pixels, labels 0-9
flat = digits.images.reshape(len(digits.images), -1)

# Stand-in embedding model: PCA to 32 dimensions, scaled to unit length.
vectors = PCA(n_components=32, svd_solver="full", random_state=0).fit_transform(flat)
vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def to_png(pixels):
    """Encode an 8x8 array of 0-16 intensities as grayscale PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(np.round(pixels * 255 / 16).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


records = [
    {
        "image_id": i,
        "label": int(digits.target[i]),  # which digit the image shows
        "split": "test" if i % 5 == 0 else "train",
        "ink": float(np.count_nonzero(digits.images[i]) / 64),  # share of inked pixels
        "image_png": to_png(digits.images[i]),
        "embedding": vectors[i].tolist(),
    }
    for i in range(len(digits.images))
]
print(len(records), "records")
```

Expected output:

```text
1797 records
```

> **About the embeddings.** A real application turns each image into a vector
> with an image model such as CLIP. This guide uses PCA over the raw pixels so it
> runs offline in seconds. PCA vectors are a weaker stand-in, but search treats
> them the same way: similar-looking images get nearby vectors. See
> [Use a real embedding model](#use-a-real-embedding-model).

## 2. Create a table

Connect to a local database directory and create a table from the records.

```python
import lancedb
import pyarrow as pa

schema = pa.schema(
    [
        pa.field("image_id", pa.int32(), nullable=False),
        pa.field("label", pa.int8()),
        pa.field("split", pa.string()),
        pa.field("ink", pa.float32()),
        pa.field("image_png", pa.binary()),  # the image itself, as bytes
        pa.field("embedding", pa.list_(pa.float32(), 32)),  # fixed-size vector
    ]
)

db = lancedb.connect("data/quickstart_db")  # creates the directory if needed
table = db.create_table(
    "digits",
    data=pa.Table.from_pylist(records, schema=schema),
    mode="overwrite",  # replaces an existing "digits" table, so you can rerun
)
print(table.count_rows(), "rows")
```

Expected output:

```text
1797 rows
```

The schema is optional, but it makes column types predictable. Two rules matter
for vector search:

- The vector column must be a **fixed-size list of floats**, here
  `pa.list_(pa.float32(), 32)`. Every row must have exactly 32 values.
- Without a schema, LanceDB infers a fixed-size vector only for list columns
  whose names contain `vector` or `embedding`. A column named `features`, for
  example, stays a variable-length list.

Images live in the table as ordinary binary values next to their vectors and
metadata. For large files such as video, use LanceDB's blob encoding instead,
which avoids loading every payload during scans; see
[Multimodal data](https://docs.lancedb.com/tables/multimodal.md).

## 3. Find similar images

Use the embedding of image 0, a handwritten 0, as the query. Ask for the five
nearest images, excluding image 0 itself.

```python
query = vectors[0].tolist()

results = (
    table.search(query, vector_column_name="embedding")
    .distance_type("cosine")
    .where("image_id != 0")
    .select(["image_id", "label", "_distance"])
    .limit(5)
    .to_list()
)
for row in results:
    print(row["image_id"], row["label"], f"{row['_distance']:.2f}")
```

Expected output (image ID, label, distance):

```text
877 0 0.05
1365 0 0.06
1167 0 0.06
1541 0 0.07
1029 0 0.07
```

All five matches are handwritten 0s. What each call does:

| Call | Purpose |
|---|---|
| `search(query, vector_column_name="embedding")` | Starts a vector search. Name the column whenever it isn't called `vector`. |
| `distance_type("cosine")` | Compares vectors by angle. Use the metric your embeddings were designed for; cosine suits unit-length vectors. |
| `where("image_id != 0")` | SQL filter applied before the vector search, so you still get 5 rows. |
| `select([...])` | Returns only these columns. `_distance` is added by the search: smaller means more similar. |
| `to_list()` | Runs the query and returns a list of dicts. |

## 4. Filter by metadata

Combine similarity with conditions on other columns. This finds the nearest
heavily inked 0s from the test split:

```python
results = (
    table.search(query, vector_column_name="embedding")
    .distance_type("cosine")
    .where("image_id != 0 AND split = 'test' AND ink >= 0.5")
    .select(["image_id", "label", "split", "ink"])
    .limit(5)
    .to_list()
)
for row in results:
    print(row["image_id"], row["label"], row["split"], row["ink"])
```

Expected output:

```text
1365 0 test 0.546875
855 0 test 0.515625
335 0 test 0.578125
725 0 test 0.5625
130 0 test 0.5
```

Filters use SQL syntax: `=`, `!=`, `<`, `>=`, `AND`, `OR`, `IN (...)`,
`LIKE`, and more. Quote strings with single quotes. See
[Metadata filtering](https://docs.lancedb.com/search/filtering.md).

## 5. Add indexes

Without an index, LanceDB compares the query with every row, which is exact and
fast for a table this size. On large tables, a vector index makes search much
faster in exchange for approximate results. Here's how to add one, plus a scalar
index for the `label` column:

```python
from lancedb.index import IvfPq

table.create_index(
    "embedding",
    config=IvfPq(distance_type="cosine", num_partitions=8, num_sub_vectors=8),
    name="embedding_idx",
)
table.create_scalar_index("label", index_type="BITMAP", name="label_idx")

print(sorted(index.name for index in table.list_indices()))
print(table.index_stats("embedding_idx").num_unindexed_rows, "unindexed rows")
```

Expected output:

```text
['embedding_idx', 'label_idx']
0 unindexed rows
```

Things to know about these calls in LanceDB 0.41:

- **Pass the column name first and an index config object.** The older form,
  `create_index(metric="cosine", vector_column_name=..., index_type=...)`, still
  works but raises a `DeprecationWarning`. So does a bare `create_index()`.
- **Build and search with the same metric.** A search with a different
  `distance_type` can't use the index and scans every row instead.
- **`num_sub_vectors` must divide the vector length evenly** (32 / 8 = 4 here).
- **The embedded library builds the index before `create_index()` returns.**
  LanceDB Enterprise builds indexes in the background; use `wait_for_index()` there.
- **IVF-PQ needs enough rows to train.** See [Troubleshooting](#troubleshooting).
- **New rows aren't indexed automatically.** After adding data, run
  `table.optimize()` to fold new rows into the index. Searches still find
  unindexed rows, more slowly, in the meantime.

To confirm a search uses the index, check its query plan:

```python
plan = (
    table.search(query, vector_column_name="embedding")
    .distance_type("cosine")
    .limit(5)
    .explain_plan()
)
print("ANN" in plan)
```

Expected output:

```text
True
```

## 6. Read images back

Images come back as the same bytes you stored. Decode them with Pillow:

```python
row = (
    table.search()
    .where("image_id = 877")
    .select(["label", "image_png"])
    .limit(1)
    .to_list()[0]
)
image = Image.open(io.BytesIO(row["image_png"]))
print(row["label"], image.size, image.mode)
```

Expected output:

```text
0 (8, 8) L
```

`table.search()` with no query runs a plain filtered scan, like a SQL `SELECT`.
Leave `image_png` out of `select()` in searches that return many rows, and fetch
image bytes only for the rows you display.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'polars'` or `'pandas'` | `.to_polars()` and `.to_pandas()` need packages that `lancedb` doesn't install. | `pip install polars` or `pip install pandas`, or use `.to_list()` / `.to_arrow()`. |
| `DeprecationWarning: The create_index() API with metric/num_partitions parameters is deprecated` | You used the older `create_index(metric=..., vector_column_name=...)` form, or `create_index()` with no arguments. | Use `create_index("column_name", config=IvfPq(distance_type="..."))`. |
| Error while training an IVF-PQ index on a small table | IVF-PQ learns its compression from your data and needs enough rows. The [probe results](../eval/results.md) record the exact message and row counts. | Skip the index on small tables (search is exact without one), or add more rows first. |
| Search on a column fails, or the column isn't treated as a vector | The column isn't a fixed-size list of floats; often its name lacks `vector`/`embedding` and no schema was given. | Declare it in the schema as `pa.list_(pa.float32(), DIM)`. |
| Search is slow after building an index | The search `distance_type` differs from the index's, so the index is skipped. | Search with the metric you indexed with. Check with `.explain_plan()`. |
| `Illegal instruction` on `import lancedb` | The default wheel needs AVX2 (x86_64 CPUs from 2013 on). | `pip uninstall lancedb && pip install lancedb-compat`. |

## Get results as a DataFrame

`to_pandas()` and `to_polars()` return DataFrames instead of dicts. Install the
library first, since neither comes with `lancedb`:

```bash
pip install pandas
```

<!-- test: skip (needs pandas, which this guide doesn't install) -->
```python
df = table.search(query, vector_column_name="embedding").distance_type("cosine").limit(5).to_pandas()
```

## Use a real embedding model

To search real photos, replace the PCA step with an image embedding model and
keep everything else. The only requirements are that every vector has the same
length and that the schema's vector size matches it. LanceDB can also compute
embeddings for you through its embedding registry; see
[OpenCLIP embeddings](https://docs.lancedb.com/integrations/embedding/openclip.md).
That path isn't covered by this guide's tests.

## For AI agents

```yaml
task: image similarity search with metadata filters in LanceDB (Python)
tested_with: {lancedb: "0.41.0", python: ["3.10", "3.12", "3.13"]}
install: pip install "lancedb==0.41.0" pillow scikit-learn
network_after_install: false
credentials: none
creates: ./data/quickstart_db (local directory)
steps: [prepare records, create table with explicit schema, vector search,
        filtered vector search, create IvfPq + bitmap indexes, read image bytes]
success_check: step 3 prints five rows, all with label 0
verification: tests/test_quickstart_doc.py runs every python block here and
              compares output with each "Expected output" block
```
