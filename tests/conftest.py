import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run a test from an empty directory so ./data paths never collide."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
