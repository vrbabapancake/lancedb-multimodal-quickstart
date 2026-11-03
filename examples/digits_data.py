"""Prepare a small, offline image dataset for the LanceDB quickstart.

This module has no LanceDB dependency, so you can inspect the data on its own.
It uses the handwritten-digits dataset that ships inside scikit-learn: 1,797
grayscale images, each 8x8 pixels, labelled 0-9. Nothing is downloaded.

Each image becomes one record with:

- the image itself, encoded as PNG bytes
- an embedding vector (EMBEDDING_DIM floats) used for similarity search
- metadata columns you can filter on with SQL

About the embeddings: a real application would use an image model such as
CLIP to turn each picture into a vector. To keep this quickstart offline,
fast and deterministic, we use PCA over the raw pixels instead. PCA vectors
are a weaker stand-in, but they behave the same way for search: images that
look alike get vectors that are close together. See "Use a real embedding
model" in docs/quickstart.md for how to swap one in.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

EMBEDDING_DIM = 32
"""Length of every embedding vector. The table schema and any index depend on it."""

TEST_EVERY = 5
"""Every 5th image (image_id divisible by 5) is assigned to the "test" split."""


def encode_png(pixels: np.ndarray) -> bytes:
    """Encode an 8x8 array of 0-16 intensities as a grayscale PNG."""
    scaled = np.round(pixels * (255.0 / 16.0)).astype(np.uint8)
    buffer = io.BytesIO()
    # A 2-D uint8 array becomes an 8-bit grayscale ("L") image.
    Image.fromarray(scaled).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(png_bytes: bytes) -> Image.Image:
    """Decode PNG bytes (as stored in the table) back into a Pillow image."""
    return Image.open(io.BytesIO(png_bytes))


def embed_images(flat_pixels: np.ndarray) -> np.ndarray:
    """Return one unit-length float32 vector of EMBEDDING_DIM values per image.

    Vectors are normalized to length 1, so cosine distance is the right
    metric for comparing them.
    """
    pca = PCA(n_components=EMBEDDING_DIM, svd_solver="full", random_state=0)
    vectors = pca.fit_transform(flat_pixels).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def make_records() -> list[dict]:
    """Build one dict per image, ready to pass to LanceDB."""
    digits = load_digits()
    images = digits.images  # shape (1797, 8, 8), values 0-16
    flat = images.reshape(len(images), -1)
    embeddings = embed_images(flat)

    records = []
    for image_id, (pixels, label, vector) in enumerate(
        zip(images, digits.target, embeddings, strict=True)
    ):
        records.append(
            {
                "image_id": image_id,
                "label": int(label),
                "split": "test" if image_id % TEST_EVERY == 0 else "train",
                # Fraction of pixels with any ink. Thicker strokes score higher.
                "ink": float(np.count_nonzero(pixels) / pixels.size),
                "image_png": encode_png(pixels),
                "embedding": vector.tolist(),
            }
        )
    return records


if __name__ == "__main__":
    rows = make_records()
    first = rows[0]
    print(f"{len(rows)} images, embedding length {len(first['embedding'])}")
    print({k: v for k, v in first.items() if k not in ("image_png", "embedding")})
