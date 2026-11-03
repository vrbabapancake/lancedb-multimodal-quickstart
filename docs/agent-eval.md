# Agent evaluation: building image search from the published LanceDB docs

An AI coding agent was given one developer task and the public LanceDB docs,
and every place it got stuck, was misled, or produced code the SDK warns about
was recorded. Each code-level finding is reproduced by a script that runs in CI
against a clean `pip install lancedb`, so anyone can check it and see when it's fixed.

**Evaluated:** November 3, 2026, against docs.lancedb.com and `lancedb` 0.41.0
(the current PyPI release). **Live results:** [eval/results.md](../eval/results.md),
refreshed by CI on every push and weekly.

## Summary

| ID | Finding | Effect on an agent | Probe result |
|---|---|---|---|
| F1 | Quickstart installs only `lancedb`, then calls `.to_polars()` | First search step crashes in a clean environment | Reproduced |
| F2, F4 | Vector index page leads with a `create_index` form the SDK deprecates; so does a bare `create_index()` | Agent writes deprecated code by default | Reproduced |
| F9 | Scalar index page teaches `create_scalar_index`, deprecated since 0.25 | Same; this evaluation's own first draft fell into it | Reproduced |
| F3 | Mixing the two `create_index` styles silently ignores `index_type` | Wrong index built, no warning or error | Reproduced |
| F6 | Index page says builds are asynchronous, with no edition caveat | Agent adds polling the embedded library doesn't need | Reproduced |
| F7 | Vector type inferred from the column *name*; error doesn't say how to fix it | Agent can't recover from the error message alone | Reproduced |
| D1 | Docs spread across four hosts; top search result fails to load | Agent reads stale or unreachable pages first | Observed |
| D2 | Quickstart's "Code" link and image path point into a docs repo readers can't use | Steps fail outside LanceDB's own checkout | Observed |
| F5 | Hypothesis: IVF-PQ fails on small tables | It doesn't: an index built on 10 rows | Not reproduced |

What works well is listed at the end; several of these are strong and worth keeping.

## Method

**Task.** "In Python, store images with embeddings and metadata in LanceDB, find
the images most similar to a query image, filter by metadata, and add an index."
Success means working code that uses current, non-deprecated APIs.

**Agent and access.** Claude, an AI coding agent, working from the published
docs as agents typically reach them: web search, then
[llms.txt](https://docs.lancedb.com/llms.txt) and the `.md` versions of pages.
It could not install packages locally, so it read the `lancedb` v0.41.0 source
to check behaviour and ran every probe in GitHub Actions.

**Evidence standard.** Behavioural claims are backed by a probe in
[eval/run_probes.py](../eval/run_probes.py), run in a fresh environment with only
`pip install lancedb`. Descriptions of what a published page says come from the
page's `.md` version, read through a tool that summarizes pages. Exact
code on those pages was confirmed with targeted questions, but check the live
page before acting on a finding.

**Limits.** One agent, one task, one day. This is a sample of how an
evaluation like this works, not a full audit.

## Findings

### F1. The quickstart's first search step needs a package it never installs

The published quickstart installs with `pip install lancedb`. Its Python search
example ends with `.to_polars()`, and `lancedb` depends on neither polars nor
pandas. In a clean environment, the probe got:

```text
to_polars: ModuleNotFoundError: No module named 'polars'
to_pandas: ModuleNotFoundError: No module named 'pandas'
```

**Why it matters for agents.** A quickstart is where an agent learns the
expected call chain. Failing at the first query costs a recovery loop, and the
fix an agent reaches for (installing a DataFrame library) adds a dependency the
task didn't need.

**Suggested fix.** End quickstart queries with `.to_list()` or `.to_arrow()`,
which need nothing extra, and show DataFrame output as an option with its install line.

**Already in progress.** On the lancedb repo's `main` branch (commit `28ba349`),
the unreleased quickstart installs `lancedb pandas`. But its Python search,
filter and feature steps are hand-written strings that call `.to_pandas()`, while
the generated snippets for those steps, the ones built from tests, still call
`.to_polars()`. The page doesn't import the generated snippets for those three
steps, so the displayed code isn't the tested code. Rendering the generated
snippets would close the gap.

### F2 and F4. The vector index page leads with a deprecated call

The page's first synchronous Python example is:

```python
table.create_index(
    metric="cosine",
    vector_column_name="keywords_embeddings",
    index_type="IVF_PQ",
)
```

In 0.41.0 this raises a `DeprecationWarning`: the SDK now takes the column name
first and an index config object. The `config=IvfPq(...)` form appears on the
page only in the async example. A bare `table.create_index()`, which the page
also presents as the way OSS users create indexes, raises the same warning: its
default first argument is the metric `"l2"`, which the SDK reads as the old form.

**Why it matters for agents.** Agents weight the first complete example on a
page heavily. Here, it trains them on an API the SDK plans to remove.

**Suggested fix.** Lead with
`table.create_index("vector", config=IvfPq(distance_type="cosine"))`, and move
the old form to a migration note. The SDK's warning already includes a clear
before-and-after example, so the docs only need to match it.

### F9. The scalar index page teaches a call deprecated since 0.25

The page's first example is `tbl.create_scalar_index("publisher", index_type="BITMAP")`.
0.41.0 warns: `create_scalar_index is deprecated as of 0.25.0. Use create_index()
with config=BTree()/Bitmap()/LabelList() instead.` The page doesn't mention the deprecation.

**This evaluation fell into it.** The first draft of this repo's quickstart used
`create_scalar_index`, written from the published docs. CI flagged the warning,
and the repo's tests now fail on LanceDB deprecation warnings. That's the strongest
evidence here that docs examples flow straight into agent-written code.

**Suggested fix.** Use `create_index("publisher", config=Bitmap())` throughout,
and run docs snippets with deprecation warnings as errors (see Recommendations).

### F3. Mixing the two `create_index` styles silently ignores `index_type`

With two styles on the same page, an agent can easily combine them. The probe ran:

```python
table.create_index("vector", index_type="IVF_HNSW_SQ")
```

LanceDB built an `IvfPq` index, not HNSW, and raised no warning or error. A
column name first selects the new style, which ignores `index_type`.

**Why it matters for agents.** Silent failures are the hardest kind for an agent:
nothing tells it to look again, and the wrong index type only shows up later as
unexpected recall or latency.

**Suggested fix.** In the SDK, warn or raise when old-style parameters such as
`index_type` arrive with a column name and no `config`. In the docs, show one
style only (F2).

### F6. "Builds asynchronously," with no edition caveat

The published index page says `create_index()` returns immediately and the index
builds asynchronously. In the embedded library, the probe found the index listed
and `num_unindexed_rows` at 0 as soon as the call returned. The asynchronous
behaviour is LanceDB Enterprise's.

**Why it matters for agents.** An agent following the page adds polling or
`wait_for_index()` calls that local code doesn't need, or treats an immediate
search as racing the build.

**Already fixed upstream.** The docs source shipped in the lancedb repo with
v0.41.0 states that the embedded library builds the index before `create_index()`
returns. The live site hasn't caught up yet.

### F7. Vector type depends on the column name, and the error doesn't say so

Without an explicit schema, LanceDB turns a list-of-floats column into a
fixed-size vector only when its name contains `vector` or `embedding`. The probe
found a column named `embedding` searchable and one named `features` not:

```text
ValueError: Invalid input, Data type is not a vector
(FixedSizeListArray or List<FixedSizeListArray>), but Float64
```

None of the pages reviewed (quickstart, multimodal, vector index, scalar index)
mention the naming rule, and the error doesn't suggest a fix.

**Suggested fix.** Document the rule where tables are created, and have the
error suggest declaring `pa.list_(pa.float32(), dim)` in the schema. An error
that names its fix lets an agent recover in one step.

### D1. Docs are spread across hosts, and the top result fails

A web search for "LanceDB documentation quickstart Python" returned results on
four hosts: lancedb.github.io, lancedb.com/documentation, www.lancedb.com/docs
and docs.lancedb.com. The top result,
`lancedb.github.io/documentation/quickstart`, failed to load. Docstrings in the
0.41.0 SDK link to `lancedb.com/docs/...`, while llms.txt lives on docs.lancedb.com.

**Suggested fix.** Redirect old hosts to docs.lancedb.com with canonical tags,
and point SDK docstrings at the same host as llms.txt.

### D2. Quickstart links and paths point into a repo readers can't use

- The quickstart's "Code" section links to `github.com/lancedb/docs/tree/main/tests`.
  This evaluation couldn't reach that repo: an anonymous clone asked for
  credentials, and GitHub reported it as not found or inaccessible. The docs
  source now appears to live in `lancedb/lancedb` under `docs/web`, with tests
  in `docs/web-tests`.
- The multimodal step reads `docs/static/assets/images/quickstart/sir-lancelot.jpg`,
  a path inside the docs repository. Outside it, the step fails with
  `FileNotFoundError`.
- The live site's structure (for example `/tables/...` in llms.txt) differs from
  the docs source in the v0.41.0 repo, which redirects `/tables` to
  `/management/table`. An agent reading the GitHub source and one reading the
  site see different page paths.

**Suggested fix.** Link to the current tests directory, and have the multimodal
step create or download its sample image.

### F5. Rejected hypothesis: IVF-PQ on small tables

The pages reviewed give no minimum row count for training an IVF-PQ index, so
this evaluation expected a failure on small tables. The probe built indexes on
10, 100, 255 and 256 rows without error. The claim was removed from this repo's
troubleshooting table before publication. It's kept here because a probe that
disproves an assumption is as useful as one that confirms it.

## What works well

- **llms.txt and `.md` pages.** Every page has a clean Markdown version, and
  llms.txt indexes about 300 of them. This is the single most useful thing for
  agent access, and many docs sites still lack it.
- **Deprecation warnings that teach.** The `create_index` warning includes a
  before-and-after code sample, exactly what an agent needs to fix its own code.
- **Explicit limits on the multimodal page.** It states which `blob_mode`
  options work where and which queries can't return blob columns.
- **Documented behaviour that holds.** F8 confirmed that searching with a
  different metric skips the index, as the index page says.
- **Tested snippets upstream.** The lancedb repo generates website code examples
  from tests and has an AGENTS.md for contributors. Most findings above are about
  pages or steps that sit outside that pipeline.

## Recommendations for an ongoing agent-evaluation program

1. **A task suite.** About 20 real developer tasks (ingest, search, filter,
   index, update, version, multimodal, RAG), each with a success check.
2. **Docs-only agents in clean containers, nightly.** Give agents llms.txt and
   nothing else, then score completion, steps to success, and deprecated or
   wrong API usage.
3. **Turn every failure into a probe.** As in this repo, each finding becomes a
   small script that fails while the problem exists and passes once it's fixed.
4. **Run snippets with deprecations as errors.** A two-line warnings filter,
   like this repo's `pyproject.toml`, catches F2, F4 and F9 before they ship.
5. **Fix the error message, not just the page.** Agents read errors more
   reliably than docs. F7 and F3 are best fixed in the SDK.
