"""End-to-end checks of examples/image_search.py against a real LanceDB table.

Vector search without an index is exact, so before indexing we compare
LanceDB's results with a brute-force NumPy computation over the same data.
"""

import numpy as np
import pytest
from PIL import Image

import image_search
from digits_data import make_records


@pytest.fixture
def table(in_tmp_dir):
    return image_search.create_table()


@pytest.fixture(scope="module")
def reference():
    """Brute-force cosine distances from image 0 to every image."""
    records = make_records()
    vectors = np.array([r["embedding"] for r in records], dtype=np.float64)
    distances = 1.0 - vectors @ vectors[0]
    return records, distances


def expected_ids(records, distances, keep, k=5):
    candidates = [i for i in np.argsort(distances, kind="stable") if keep(records[i])]
    return candidates[:k]


def test_table_schema_and_row_count(table):
    assert table.count_rows() == 1797
    actual = [(f.name, f.type) for f in table.schema]
    assert actual == [(f.name, f.type) for f in image_search.SCHEMA]


def test_vector_search_matches_brute_force(table, reference):
    records, distances = reference
    rows = image_search.find_similar(
        table, image_search.get_embedding(table, 0), where="image_id != 0"
    )
    want = expected_ids(records, distances, keep=lambda r: r["image_id"] != 0)
    assert [r["image_id"] for r in rows] == want
    for row in rows:
        assert row["_distance"] == pytest.approx(distances[row["image_id"]], abs=1e-4)
    assert all(r["label"] == 0 for r in rows)


def test_filtered_search_respects_filter(table, reference):
    records, distances = reference
    where = "image_id != 0 AND split = 'test' AND ink >= 0.5"
    rows = image_search.find_similar(table, image_search.get_embedding(table, 0), where=where)

    def keep(r):
        return r["image_id"] != 0 and r["split"] == "test" and r["ink"] >= 0.5

    assert [r["image_id"] for r in rows] == expected_ids(records, distances, keep)
    assert all(r["split"] == "test" and r["ink"] >= 0.5 for r in rows)


def test_indexes_are_built_and_used(table):
    image_search.build_indexes(table)
    indexes = {index.name: index for index in table.list_indices()}
    assert image_search.VECTOR_INDEX_NAME in indexes
    assert indexes[image_search.VECTOR_INDEX_NAME].columns == ["embedding"]
    assert "PQ" in str(indexes[image_search.VECTOR_INDEX_NAME].index_type).upper()

    stats = table.index_stats(image_search.VECTOR_INDEX_NAME)
    assert stats.num_unindexed_rows == 0

    query = image_search.get_embedding(table, 0)
    plan = (
        table.search(query, vector_column_name="embedding")
        .distance_type("cosine")
        .limit(5)
        .explain_plan()
    )
    assert "ANN" in plan.upper() or "IVF" in plan.upper(), plan

    # An approximate index can reorder neighbours, but the matches should
    # still be handwritten zeros.
    rows = image_search.find_similar(table, query, where="image_id != 0")
    assert len(rows) == 5
    assert sum(r["label"] == 0 for r in rows) >= 4


def test_results_picture(table):
    rows = image_search.find_similar(
        table, image_search.get_embedding(table, 0), where="image_id != 0"
    )
    path = image_search.save_results_picture(table, 0, rows)
    with Image.open(path) as picture:
        assert picture.size == (6 * (96 + 12) + 12, 96 + 24)


def test_main_runs(in_tmp_dir, capsys):
    image_search.main()
    out = capsys.readouterr().out
    assert "Created table 'digits' with 1797 rows" in out
    assert "index embedding_idx" in out
    assert (in_tmp_dir / "data" / "results.png").exists()
