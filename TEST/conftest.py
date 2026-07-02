"""Pytest fixtures for the ShiftDx smoke suite.

Tests live under TEST/ (never beside source). Run with `pytest TEST/` from the
repo root.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> str:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def data_dir(repo_root) -> str:
    return os.path.join(repo_root, "data")


@pytest.fixture(scope="session")
def store(data_dir):
    from data_loader import DataStore
    return DataStore(data_dir)
