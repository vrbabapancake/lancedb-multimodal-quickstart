"""Checks on the dataset itself. These need no LanceDB install."""

import numpy as np

from digits_data import EMBEDDING_DIM, decode_png, make_records


def test_records_have_expected_shape():
    records = make_records()
    assert len(records) == 1797
    first = records[0]
    assert set(first) == {"image_id", "label", "split", "ink", "image_png", "embedding"}
    assert len(first["embedding"]) == EMBEDDING_DIM
    assert sum(r["split"] == "test" for r in records) == 360


def test_embeddings_are_unit_length_and_meaningful():
    records = make_records()
    vectors = np.array([r["embedding"] for r in records])
    labels = np.array([r["label"] for r in records])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    # Nearby vectors should mostly share a label, or similarity search is useless.
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    top5 = np.argsort(-similarity, axis=1)[:, :5]
    assert (labels[top5] == labels[:, None]).mean() > 0.95


def test_png_round_trip():
    record = make_records()[0]
    image = decode_png(record["image_png"])
    assert image.size == (8, 8)
    assert image.mode == "L"
