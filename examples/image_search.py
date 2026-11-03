"""Multimodal image search with LanceDB, end to end.

Stores images, their embeddings and their metadata in one LanceDB table, then:

1. finds the images most similar to a query image (vector search)
2. narrows the search with a SQL filter on metadata
3. builds a vector index and a scalar index, and searches through them
4. reads stored image bytes back and saves a picture of the results

Run it from the repository root:

    pip install -r requirements.txt
    python examples/image_search.py

It creates a local database in ./data/digits_db (deleted and rebuilt on every
run) and writes the results picture to ./data/results.png. No account, API key
or network access is needed after installing packages.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import IvfPq
from PIL import Image

from digits_data import EMBEDDING_DIM, decode_png, make_records

DB_PATH = Path("data/digits_db")
TABLE_NAME = "digits"
VECTOR_INDEX_NAME = "embedding_idx"

# An explicit schema makes column types predictable. Without one, LanceDB infers
# types from the data: Python ints become int64, and a list column becomes a
# fixed-size vector only if its name contains "vector" or "embedding".
SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.int32(), nullable=False),
        pa.field("label", pa.int8()),
        pa.field("split", pa.string()),
        pa.field("ink", pa.float32()),
        pa.field("image_png", pa.binary()),
        # Vector columns must be fixed-size lists of floats.
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ]
)

# Columns to return from searches. Leaving out image_png and embedding keeps
# result sets small; fetch the image bytes only for the rows you display.
RESULT_COLUMNS = ["image_id", "label", "split", "ink", "_distance"]


def create_table(db_path: Path = DB_PATH) -> lancedb.table.Table:
    """Create (or recreate) the digits table and load every image into it."""
    shutil.rmtree(db_path, ignore_errors=True)
    db = lancedb.connect(db_path)
    records = make_records()
    return db.create_table(
        TABLE_NAME,
        data=pa.Table.from_pylist(records, schema=SCHEMA),
        mode="overwrite",
    )


def get_embedding(table: lancedb.table.Table, image_id: int) -> list[float]:
    """Look up the stored embedding of one image, to use as a query."""
    rows = table.search().where(f"image_id = {image_id}").select(["embedding"]).limit(1).to_list()
    if not rows:
        raise KeyError(f"No image with image_id = {image_id}")
    return rows[0]["embedding"]


def find_similar(
    table: lancedb.table.Table,
    query_vector: list[float],
    k: int = 5,
    where: str | None = None,
) -> list[dict]:
    """Return the k nearest images to query_vector, optionally filtered by SQL.

    The embeddings are unit-length, so we compare them by cosine distance.
    Pass the same metric the vector index was built with; a different metric
    cannot use the index and falls back to a full scan.
    """
    query = (
        table.search(query_vector, vector_column_name="embedding")
        .distance_type("cosine")
        .select(RESULT_COLUMNS)
        .limit(k)
    )
    if where:
        # Filters run before the vector search by default (prefilter=True),
        # so you always get k matching rows when at least k exist.
        query = query.where(where)
    return query.to_list()


def build_indexes(table: lancedb.table.Table) -> None:
    """Add an IVF-PQ vector index on embedding and a scalar index on label.

    A table this small does not need a vector index: without one, LanceDB
    compares the query with every row and returns exact results quickly. An
    index trades a little accuracy for speed on large tables. We build one
    here on 1,797 rows only to show the calls.
    """
    table.create_index(
        "embedding",
        config=IvfPq(
            distance_type="cosine",
            num_partitions=8,  # about 225 rows per partition for 1,797 rows
            num_sub_vectors=8,  # must divide EMBEDDING_DIM (32) evenly
        ),
        name=VECTOR_INDEX_NAME,
    )
    # A bitmap index suits low-cardinality columns such as a 0-9 label.
    table.create_scalar_index("label", index_type="BITMAP")


def save_results_picture(
    table: lancedb.table.Table,
    query_id: int,
    results: list[dict],
    out_path: Path = Path("data/results.png"),
    scale: int = 12,
) -> Path:
    """Fetch stored PNG bytes and save the query image beside its matches."""
    ids = [query_id] + [row["image_id"] for row in results]
    id_list = ", ".join(str(i) for i in ids)
    rows = (
        table.search()
        .where(f"image_id IN ({id_list})")
        .select(["image_id", "image_png"])
        .limit(len(ids))
        .to_list()
    )
    png_by_id = {row["image_id"]: row["image_png"] for row in rows}

    tile = 8 * scale
    gap = scale
    sheet = Image.new("L", (len(ids) * (tile + gap) + gap, tile + 2 * gap), 255)
    for position, image_id in enumerate(ids):
        image = decode_png(png_by_id[image_id]).resize((tile, tile), Image.NEAREST)
        sheet.paste(image, (gap + position * (tile + gap), gap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def print_rows(rows: list[dict]) -> None:
    print(f"  {'image_id':>8}  {'label':>5}  {'split':<5}  {'ink':>5}  {'distance':>8}")
    for row in rows:
        print(
            f"  {row['image_id']:>8}  {row['label']:>5}  {row['split']:<5}"
            f"  {row['ink']:>5.2f}  {row['_distance']:>8.4f}"
        )


def main() -> None:
    table = create_table()
    print(f"Created table '{TABLE_NAME}' with {table.count_rows()} rows")

    query_id = 0  # a handwritten 0
    query_vector = get_embedding(table, query_id)

    print("\nStep 1: five images most similar to image 0 (excluding itself)")
    similar = find_similar(table, query_vector, where=f"image_id != {query_id}")
    print_rows(similar)

    print("\nStep 2: same search, only test-split images with heavy ink")
    filtered = find_similar(
        table,
        query_vector,
        where=f"image_id != {query_id} AND split = 'test' AND ink >= 0.5",
    )
    print_rows(filtered)

    print("\nStep 3: build indexes, then search again")
    build_indexes(table)
    for index in table.list_indices():
        print(f"  index {index.name}: {index.index_type} on {index.columns}")
    indexed = find_similar(table, query_vector, where=f"image_id != {query_id}")
    print_rows(indexed)

    picture = save_results_picture(table, query_id, similar)
    print(f"\nStep 4: saved query image and matches to {picture}")


if __name__ == "__main__":
    main()
