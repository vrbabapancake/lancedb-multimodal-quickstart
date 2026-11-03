# LanceDB multimodal quickstart, tested and agent-ready

[![CI](https://github.com/vrbabapancake/lancedb-multimodal-quickstart/actions/workflows/ci.yml/badge.svg)](https://github.com/vrbabapancake/lancedb-multimodal-quickstart/actions/workflows/ci.yml)

A documentation-engineering sample by [Mark Phillips](https://vrbabapancake.github.io/marks-repo/).
It has three parts:

| Part | What it is |
|---|---|
| [Quickstart: Find similar images with LanceDB](docs/quickstart.md) | A tutorial for storing images, embeddings and metadata in one LanceDB table, then searching, filtering and indexing. Runs offline in about 10 minutes. |
| [Runnable example](examples/image_search.py) | The same workflow as a script, with tests that check its results against a brute-force NumPy computation. |
| [Agent evaluation of the LanceDB docs](docs/agent-eval.md) | What happened when an AI coding agent tried to complete this task from the published LanceDB docs, with each finding reproduced by a script in CI. |

## How the docs stay accurate

- **Every code block in the quickstart runs in CI.** [`tests/test_quickstart_doc.py`](tests/test_quickstart_doc.py)
  executes the guide's Python blocks in order and fails if any output differs
  from the guide's "Expected output" blocks.
- **Deprecated calls fail the build.** Tests turn LanceDB's
  deprecation warnings into errors, so the guide can't teach a call the SDK is retiring.
- **Pinned and latest.** CI tests the pinned `lancedb==0.41.0` on Linux and macOS
  with Python 3.10 to 3.13, and runs weekly against the newest LanceDB release to catch drift.
- **Findings are reproducible.** [`eval/run_probes.py`](eval/run_probes.py) re-runs
  each documentation finding in a clean environment and reports whether it still occurs.

## Run it

```bash
pip install -r requirements.txt
python examples/image_search.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## For AI agents

Start with [`AGENTS.md`](AGENTS.md) or [`llms.txt`](llms.txt).
