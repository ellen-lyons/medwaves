"""Leakage tests. Keep these: held-out buoys and the held-out mission must never reach model inputs."""
import datetime as dt
from pathlib import Path

import pytest
import yaml

from medwaves.config import Paths, config_from_dict
from medwaves.splits import (
    ALTIMETER,
    BUOY,
    INPUT,
    SPLIT_NAMES,
    TARGET,
    LeakageError,
    ObsRole,
    SpaceSplit,
    TimeSplit,
    draft_buoy_split_path,
    obs_store,
    write_draft_buoy_split,
)


@pytest.fixture
def time_split(raw_config):
    return TimeSplit.from_config(config_from_dict(raw_config))


@pytest.fixture
def space():
    return SpaceSplit(
        train_buoys=frozenset({"B1", "B2", "B3"}),
        heldout_buoys=frozenset({"H1", "H2"}),
        train_missions=frozenset({"mission_a"}),
        heldout_mission="mission_b",
    )


# Time split

def test_windows_match_brief(time_split):
    w = time_split.windows
    assert (w["train"].start, w["train"].end) == (dt.datetime(2019, 1, 1), dt.datetime(2022, 1, 1))
    assert (w["val"].start, w["val"].end) == (dt.datetime(2022, 1, 8), dt.datetime(2023, 1, 1))
    assert (w["test"].start, w["test"].end) == (dt.datetime(2023, 1, 8), dt.datetime(2024, 1, 1))


def test_windows_disjoint_with_gap(time_split, raw_config):
    gap = dt.timedelta(days=raw_config["splits"]["gap_days"])
    windows = [time_split.windows[n] for n in SPLIT_NAMES]
    for earlier, later in zip(windows, windows[1:]):
        assert later.start - earlier.end >= gap


def test_which(time_split):
    assert time_split.which(dt.datetime(2020, 6, 1)) == "train"
    assert time_split.which(dt.datetime(2022, 1, 3)) == "gap"
    assert time_split.which(dt.datetime(2022, 6, 1)) == "val"
    assert time_split.which(dt.datetime(2023, 1, 3)) == "gap"
    assert time_split.which(dt.datetime(2023, 12, 31, 21)) == "test"
    assert time_split.which(dt.datetime(2024, 1, 1)) == "gap"


def test_sample_history_must_stay_inside_split(time_split):
    start = time_split.windows["val"].start
    assert not time_split.sample_ok(start, "val")                    # history reaches into the gap
    assert not time_split.sample_ok(start + dt.timedelta(hours=21), "val")
    assert time_split.sample_ok(start + dt.timedelta(hours=24), "val")
    assert not time_split.sample_ok(dt.datetime(2022, 6, 1), "train")


def test_no_valid_sample_has_history_in_another_split(time_split):
    t = dt.datetime(2019, 1, 1)
    step = dt.timedelta(hours=3)
    while t < dt.datetime(2024, 1, 1):
        for name in SPLIT_NAMES:
            if time_split.sample_ok(t, name):
                assert time_split.which(t - time_split.wind_history) == name
                assert time_split.which(t - time_split.obs_history) == name
        t += step


def test_observation_lookback_is_checked_on_its_own(raw_config):
    raw_config["time"]["obs_history_hours"] = 72      # longer than the 24 h wind history
    ts = TimeSplit.from_config(config_from_dict(raw_config))
    start = ts.windows["val"].start
    assert not ts.sample_ok(start + dt.timedelta(hours=48), "val")   # wind fits, obs doesn't
    assert ts.sample_ok(start + dt.timedelta(hours=72), "val")


def test_obs_input_must_be_inside_lookback_and_split(time_split):
    t = dt.datetime(2022, 6, 1, 12)
    assert time_split.obs_input_ok(t - dt.timedelta(hours=6), t, "val")
    assert not time_split.obs_input_ok(t - dt.timedelta(hours=30), t, "val")   # too old
    assert not time_split.obs_input_ok(t + dt.timedelta(hours=3), t, "val")    # from the future
    first = time_split.windows["val"].start + dt.timedelta(hours=24)
    assert time_split.obs_input_ok(time_split.windows["val"].start, first, "val")
    assert not time_split.obs_input_ok(time_split.windows["val"].start - dt.timedelta(hours=3),
                                       first, "val")                          # in the gap


# Space split

def test_roles(space):
    assert space.role(BUOY, "B1") is ObsRole.TRAIN_POOL
    assert space.role(BUOY, "H1") is ObsRole.HELDOUT
    assert space.role(ALTIMETER, "mission_a") is ObsRole.TRAIN_POOL
    assert space.role(ALTIMETER, "mission_b") is ObsRole.HELDOUT


def test_unknown_platforms_fail_closed(space):
    with pytest.raises(LeakageError):
        space.role(BUOY, "NEW")
    with pytest.raises(LeakageError):
        space.role(ALTIMETER, "mission_c")


def test_default_config_missions(raw_config):
    split = SpaceSplit.from_config(config_from_dict(raw_config))
    assert split.role(ALTIMETER, raw_config["pilot"]["mission"]) is ObsRole.TRAIN_POOL
    assert split.role(ALTIMETER, raw_config["splits"]["space"]["heldout_mission"]) is ObsRole.HELDOUT


def test_unfrozen_buoy_split_refuses_every_buoy(raw_config):
    raw_config["splits"]["space"]["heldout_buoys_file"] = "configs/does_not_exist.yaml"
    split = SpaceSplit.from_config(config_from_dict(raw_config))
    assert not split.buoys_frozen
    with pytest.raises(LeakageError):
        split.role(BUOY, "B1")


def test_overlapping_buoy_sets_rejected():
    with pytest.raises(LeakageError):
        SpaceSplit(frozenset({"B1"}), frozenset({"B1"}), frozenset(), None)


def test_heldout_mission_cannot_be_train_mission():
    with pytest.raises(LeakageError):
        SpaceSplit(frozenset(), frozenset(), frozenset({"m"}), "m")


def test_assert_no_heldout(space):
    space.assert_no_heldout(["B1", "B2"], ["mission_a"])
    with pytest.raises(LeakageError, match="H2"):
        space.assert_no_heldout(["B1", "H2"], ["mission_a"])
    with pytest.raises(LeakageError, match="mission_b"):
        space.assert_no_heldout(["B1"], ["mission_a", "mission_b"])
    with pytest.raises(LeakageError):
        space.assert_no_heldout(["UNKNOWN"])


# Draft buoy split (written to Drive, copied into the repo by hand)

def test_draft_written_once_and_never_redrawn(raw_config, tmp_path):
    raw_config["splits"]["space"]["heldout_buoys_file"] = "configs/does_not_exist.yaml"
    cfg = config_from_dict(raw_config, tmp_path)
    draft = write_draft_buoy_split(cfg, ["B2", "B1"], ["H1"], seed=1, inventory_hash="abc")
    assert draft == tmp_path / "splits" / "does_not_exist.yaml"
    saved = yaml.safe_load(draft.read_text())
    assert saved["train_pool"] == ["B1", "B2"] and saved["heldout"] == ["H1"]
    with pytest.raises(FileExistsError):
        write_draft_buoy_split(cfg, ["B1"], ["H1", "H2"], seed=2, inventory_hash="def")
    # The draft is not active until committed to the repo.
    assert not SpaceSplit.from_config(cfg).buoys_frozen


def test_draft_refused_when_repo_split_exists(raw_config, tmp_path):
    raw_config["splits"]["space"]["heldout_buoys_file"] = "configs/default.yaml"  # any existing file
    cfg = config_from_dict(raw_config, tmp_path)
    with pytest.raises(FileExistsError):
        write_draft_buoy_split(cfg, ["B1"], ["H1"], seed=1, inventory_hash="abc")


def test_draft_with_overlap_rejected(raw_config, tmp_path):
    raw_config["splits"]["space"]["heldout_buoys_file"] = "configs/does_not_exist.yaml"
    cfg = config_from_dict(raw_config, tmp_path)
    with pytest.raises(LeakageError):
        write_draft_buoy_split(cfg, ["B1"], ["B1"], seed=1, inventory_hash="abc")
    assert not draft_buoy_split_path(cfg).exists()


# Stores

def test_stores_read_from_local_copy_when_set():
    paths = Paths(Path("/drive"), local_root=Path("/content/local"))
    assert obs_store(paths, ObsRole.TRAIN_POOL, INPUT) == Path("/content/local/processed/obs/train_pool")


def test_heldout_store_cannot_be_read_as_input():
    paths = Paths(Path("/data"))
    assert obs_store(paths, ObsRole.TRAIN_POOL, INPUT) == Path("/data/processed/obs/train_pool")
    assert obs_store(paths, ObsRole.HELDOUT, TARGET) == Path("/data/processed/obs/heldout")
    with pytest.raises(LeakageError):
        obs_store(paths, ObsRole.HELDOUT, INPUT)
    with pytest.raises(LeakageError):
        obs_store(paths, "heldout", INPUT)
