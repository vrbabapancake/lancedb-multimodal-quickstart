# Instructions for AI coding agents

This repository is a tested LanceDB tutorial. Use it to learn the LanceDB
Python API for multimodal similarity search, or to change the tutorial.

## Facts to rely on

- Tested with `lancedb==0.41.0`, Python 3.10 to 3.13. Install with
  `pip install -r requirements-dev.txt`. No network access, account or API key
  is needed after install.
- Read query results with `.to_list()` or `.to_arrow()`. `.to_pandas()` and
  `.to_polars()` need `pandas` or `polars`, which `lancedb` doesn't install.
- Create vector indexes with the column name first and a config object:
  `table.create_index("embedding", config=IvfPq(distance_type="cosine"))`.
  The `create_index(metric=..., vector_column_name=...)` form is deprecated, and
  this repo's tests fail if you use it.
- Create scalar indexes the same way: `create_index("label", config=Bitmap())`.
  `create_scalar_index()` is deprecated since LanceDB 0.25.
- Search with the same `distance_type` the index was built with, or the index is
  skipped. Confirm with `.explain_plan()`.
- Declare vector columns as `pa.list_(pa.float32(), DIM)` in an explicit schema.

## Making changes

- Keep `docs/quickstart.md` runnable top to bottom: `tests/test_quickstart_doc.py`
  executes every Python block and compares printed output with each
  "Expected output" block. Update both together.
- Mark a block that can't run in CI with `<!-- test: skip (reason) -->` on the line before it.
- Run `ruff check .` and `pytest` before committing.
- Don't add claims about LanceDB behaviour to the docs unless a test or an
  `eval/run_probes.py` probe demonstrates them.
