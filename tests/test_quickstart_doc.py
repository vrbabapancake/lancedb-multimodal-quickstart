"""Run docs/quickstart.md as a test.

Every ```python block runs in order in one shared namespace, the way a reader
would run them. When a block is followed by an "Expected output" paragraph and
a ```text block, the printed output must match it exactly.

Put `<!-- test: skip ... -->` on the line before a block to leave it out.
"""

import contextlib
import io
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)


def parse_blocks(text):
    """Yield (code, expected_output_or_None) for each runnable python block."""
    fences = list(FENCE.finditer(text))
    for index, fence in enumerate(fences):
        if fence.group(1) != "python":
            continue
        before = text[: fence.start()].rstrip().splitlines()
        if before and before[-1].strip().startswith("<!-- test: skip"):
            continue
        expected = None
        if index + 1 < len(fences) and fences[index + 1].group(1) == "text":
            between = text[fence.end() : fences[index + 1].start()]
            if between.strip().lower().startswith("expected output"):
                expected = fences[index + 1].group(2)
        yield fence.group(2), expected


def test_quickstart_runs_and_matches_expected_output(in_tmp_dir):
    blocks = list(parse_blocks(DOC.read_text()))
    assert len(blocks) >= 7, "quickstart lost some of its code blocks"

    namespace = {"__name__": "__quickstart__"}
    for number, (code, expected) in enumerate(blocks, start=1):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, f"quickstart block {number}", "exec"), namespace)
        if expected is not None:
            assert stdout.getvalue().strip() == expected.strip(), (
                f"Block {number} printed:\n{stdout.getvalue()}\nbut the doc says:\n{expected}"
            )
