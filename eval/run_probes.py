"""Reproduce the agent-evaluation findings against the released lancedb package.

Each probe follows what a published docs page tells a reader (or an AI agent)
to do, runs it, and records what actually happens. Probes never fail the
build: they report REPRODUCED when the documented friction still occurs and
NOT REPRODUCED when it doesn't (for example, after a docs or SDK fix).

CI runs this in a fresh environment with only `pip install lancedb`, exactly
as the published quickstart instructs. Results go to eval/results.md and
eval/results.json, and to the GitHub Actions job summary.

    python eval/run_probes.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import random
import tempfile
import traceback
import warnings
from pathlib import Path

import lancedb
import pyarrow as pa

HERE = Path(__file__).resolve().parent
DIM = 16
PROBES = []


def probe(finding: str, title: str):
    def register(fn):
        PROBES.append((finding, title, fn))
        return fn

    return register


def make_table(db_path: Path, rows: int, column: str = "vector", name: str = "t"):
    """Create a table of random unit-ish vectors, inferring the schema."""
    rng = random.Random(0)
    data = [
        {"id": i, "label": i % 10, column: [rng.random() for _ in range(DIM)]} for i in range(rows)
    ]
    db = lancedb.connect(db_path)
    return db.create_table(name, data=data, mode="overwrite")


def short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text if len(text) <= 300 else text[:297] + "..."


@probe("F1", "Quickstart search step needs a package the install step leaves out")
def quickstart_dataframe_dependency(tmp: Path) -> dict:
    table = make_table(tmp / "db", rows=10)
    query = [0.5] * DIM
    observed = {}
    for method in ("to_polars", "to_pandas"):
        try:
            getattr(table.search(query).limit(2), method)()
            observed[method] = "worked"
        except Exception as exc:  # noqa: BLE001
            observed[method] = short_error(exc)
    reproduced = observed["to_polars"] != "worked"
    return {
        "reproduced": reproduced,
        "observed": observed,
        "installed": {
            "polars": importlib.util.find_spec("polars") is not None,
            "pandas": importlib.util.find_spec("pandas") is not None,
        },
        "docs_say": "Install with `pip install lancedb`; the Python search example "
        "ends in `.to_polars()`.",
    }


@probe("F2", "Vector index page teaches a create_index call the SDK deprecates")
def legacy_create_index(tmp: Path) -> dict:
    table = make_table(tmp / "db", rows=512, column="keywords_embeddings")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Exactly the first synchronous Python example on the published page.
        table.create_index(
            metric="cosine",
            vector_column_name="keywords_embeddings",
            index_type="IVF_PQ",
        )
    deprecations = [
        str(w.message).splitlines()[0] for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    return {
        "reproduced": bool(deprecations),
        "observed": deprecations or "no DeprecationWarning",
        "docs_say": "table.create_index(metric='cosine', "
        "vector_column_name='keywords_embeddings', index_type='IVF_PQ')",
    }


@probe("F3", "Mixing old and new create_index styles: what happens to index_type?")
def mixed_style_create_index(tmp: Path) -> dict:
    table = make_table(tmp / "db", rows=512)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Column name first (new style) plus index_type (old style): a call an
        # agent can easily assemble from the two styles shown in the docs.
        table.create_index("vector", index_type="IVF_HNSW_SQ")
    built = [(i.name, str(i.index_type)) for i in table.list_indices()]
    built_types = " ".join(t for _, t in built).upper()
    return {
        "reproduced": "HNSW" not in built_types,
        "observed": {
            "indexes": built,
            "warnings": [str(w.message).splitlines()[0] for w in caught],
        },
        "docs_say": "index_type selects the index type (e.g. IVF_HNSW_SQ).",
    }


@probe("F4", "No-argument create_index() raises the legacy-API DeprecationWarning")
def bare_create_index(tmp: Path) -> dict:
    table = make_table(tmp / "db", rows=512)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table.create_index()
    deprecations = [
        str(w.message).splitlines()[0] for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    return {
        "reproduced": bool(deprecations),
        "observed": deprecations or "no DeprecationWarning",
        "docs_say": "OSS users can create vector indexes by calling table.create_index().",
    }


@probe("F5", "Too few rows to train an index: the docs give no minimum")
def too_few_rows(tmp: Path) -> dict:
    from lancedb.index import IvfPq

    observed = {}
    for rows in (10, 100, 255, 256):
        table = make_table(tmp / f"db{rows}", rows=rows)
        try:
            table.create_index("vector", config=IvfPq())
            observed[rows] = "index built"
        except Exception as exc:  # noqa: BLE001
            observed[rows] = short_error(exc)
    failed = [r for r, v in observed.items() if v != "index built"]
    return {
        "reproduced": bool(failed),
        "observed": observed,
        "docs_say": "No minimum row count or error is documented.",
    }


@probe("F6", "create_index() in embedded LanceDB: blocking or asynchronous?")
def index_build_timing(tmp: Path) -> dict:
    from lancedb.index import IvfPq

    table = make_table(tmp / "db", rows=2048)
    table.create_index("vector", config=IvfPq(), name="vector_idx")
    listed = [i.name for i in table.list_indices()]
    stats = table.index_stats("vector_idx")
    ready = "vector_idx" in listed and stats is not None and stats.num_unindexed_rows == 0
    return {
        # The published page says the build is asynchronous with no edition
        # caveat; it is reproduced as a doc gap when the index is ready at once.
        "reproduced": ready,
        "observed": {
            "listed_immediately": listed,
            "num_unindexed_rows": None if stats is None else stats.num_unindexed_rows,
        },
        "docs_say": "create_index() returns immediately, but the vector index "
        "builds asynchronously.",
    }


@probe("F7", "Vector type is inferred from the column name")
def vector_name_inference(tmp: Path) -> dict:
    observed = {}
    for column in ("embedding", "features"):
        table = make_table(tmp / f"db_{column}", rows=20, column=column)
        field_type = str(table.schema.field(column).type)
        try:
            table.search([0.5] * DIM, vector_column_name=column).limit(1).to_arrow()
            result = "search worked"
        except Exception as exc:  # noqa: BLE001
            result = short_error(exc)
        observed[column] = {"type": field_type, "search": result}
    reproduced = observed["features"]["search"] != "search worked"
    return {
        "reproduced": reproduced,
        "observed": observed,
        "docs_say": "Not covered on the vector search or ingestion pages reviewed.",
    }


@probe("F8", "Searching with a different metric than the index was built with")
def metric_mismatch(tmp: Path) -> dict:
    from lancedb.index import IvfPq

    table = make_table(tmp / "db", rows=2048)
    table.create_index("vector", config=IvfPq(distance_type="cosine"))
    query = [0.5] * DIM
    plans = {}
    for label, builder in {
        "omitted": table.search(query),
        "cosine": table.search(query).distance_type("cosine"),
        "l2": table.search(query).distance_type("l2"),
    }.items():
        plan = builder.limit(5).explain_plan().upper()
        plans[label] = "uses index (ANN)" if "ANN" in plan else "full scan (no ANN node)"
    return {
        "reproduced": plans["l2"] != "uses index (ANN)",
        "observed": plans,
        "docs_say": "A different metric cannot use the index and falls back to a flat search.",
    }


def main() -> None:
    results = []
    for finding, title, fn in PROBES:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                outcome = fn(Path(tmp))
            except Exception:  # noqa: BLE001
                outcome = {"reproduced": None, "observed": traceback.format_exc(limit=3)}
        outcome.update(finding=finding, title=title)
        results.append(outcome)
        status = {True: "REPRODUCED", False: "NOT REPRODUCED", None: "PROBE ERROR"}[
            outcome["reproduced"]
        ]
        print(f"{finding} {status}: {title}")

    env = {
        "lancedb": lancedb.__version__,
        "pyarrow": pa.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (HERE / "results.json").write_text(
        json.dumps({"environment": env, "results": results}, indent=2, default=str)
    )

    lines = [
        "# Probe results",
        "",
        f"lancedb {env['lancedb']}, pyarrow {env['pyarrow']}, "
        f"Python {env['python']} on {env['platform']}",
        "",
        "| Finding | Status | Probe |",
        "|---|---|---|",
    ]
    for r in results:
        status = {True: "Reproduced", False: "Not reproduced", None: "Probe error"}[r["reproduced"]]
        lines.append(f"| {r['finding']} | {status} | {r['title']} |")
    lines += ["", "## Details", ""]
    for r in results:
        lines += [
            f"### {r['finding']}: {r['title']}",
            "",
            f"- Docs say: {r.get('docs_say', 'n/a')}",
            "- Observed:",
            "",
            "```json",
            json.dumps(r["observed"], indent=2, default=str),
            "```",
            "",
        ]
    (HERE / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
