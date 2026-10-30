import copy

import pytest
import yaml

from medwaves.config import DATA_ROOT_ENV, DEFAULT_CONFIG, LOCAL_ROOT_ENV


@pytest.fixture
def raw_config(monkeypatch):
    """A fresh copy of the default config as parsed YAML, with no path overrides from the env."""
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(LOCAL_ROOT_ENV, raising=False)
    with open(DEFAULT_CONFIG) as f:
        return copy.deepcopy(yaml.safe_load(f))
