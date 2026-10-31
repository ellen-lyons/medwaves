from pathlib import Path

import pytest

from medwaves.config import (
    DATA_ROOT_ENV,
    LOCAL_ROOT_ENV,
    ConfigError,
    config_from_dict,
    load_config,
)


def test_default_config_loads(raw_config):
    cfg = config_from_dict(raw_config)
    assert cfg.model.latent == 64
    assert cfg.paths.data_root == Path("/content/drive/MyDrive/medwaves")
    assert cfg.products["wave_reanalysis"].role == "referee_only"


def test_load_config_reads_default_file(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    assert load_config().seed == 1234


def test_data_root_priority(raw_config, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, "/from/env")
    assert config_from_dict(raw_config).paths.data_root == Path("/from/env")
    assert config_from_dict(raw_config, "/from/arg").paths.data_root == Path("/from/arg")


def test_unknown_key_rejected(raw_config):
    raw_config["model"]["hidden"] = 128
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict(raw_config)


def test_missing_key_rejected(raw_config):
    del raw_config["splits"]["gap_days"]
    with pytest.raises(ConfigError):
        config_from_dict(raw_config)


def test_wrong_type_rejected(raw_config):
    raw_config["model"]["latent"] = "64"
    with pytest.raises(ConfigError, match="expected int"):
        config_from_dict(raw_config)


@pytest.mark.parametrize("role", ["observation", "input", "target"])
def test_reanalysis_must_be_referee_only(raw_config, role):
    raw_config["products"]["wave_reanalysis"]["role"] = role
    with pytest.raises(ConfigError, match="referee_only"):
        config_from_dict(raw_config)


def test_overlapping_splits_rejected(raw_config):
    raw_config["splits"]["val"]["start"] = "2021-06-01"
    with pytest.raises(ConfigError, match="overlaps"):
        config_from_dict(raw_config)


def test_heldout_mission_cannot_be_a_train_mission(raw_config):
    raw_config["splits"]["space"]["heldout_mission"] = "m1"
    raw_config["splits"]["space"]["train_missions"] = ["m1", "m2"]
    with pytest.raises(ConfigError, match="heldout_mission"):
        config_from_dict(raw_config)


def test_pilot_mission_cannot_be_heldout_mission(raw_config):
    raw_config["pilot"]["mission"] = raw_config["splits"]["space"]["heldout_mission"]
    with pytest.raises(ConfigError, match="pilot.mission"):
        config_from_dict(raw_config)


def test_pilot_mission_must_be_a_train_mission(raw_config):
    raw_config["pilot"]["mission"] = "unlisted"
    with pytest.raises(ConfigError, match="pilot.mission"):
        config_from_dict(raw_config)


def test_local_root_from_env(raw_config, monkeypatch):
    monkeypatch.setenv(LOCAL_ROOT_ENV, "/content/local")
    cfg = config_from_dict(raw_config)
    assert cfg.paths.processed_read == Path("/content/local/processed")
    assert cfg.paths.processed == Path("/content/drive/MyDrive/medwaves/processed")


def test_unverified_product_refused(raw_config):
    raw_config["products"]["altimeter_l3_my"]["verified"] = False
    cfg = config_from_dict(raw_config)
    with pytest.raises(ConfigError, match="not verified"):
        cfg.require_verified("altimeter_l3_my")


def test_verified_product_returned(raw_config):
    cfg = config_from_dict(raw_config)
    my = cfg.require_verified("altimeter_l3_my")
    assert my.swh_variable == "swh_adjusted"
    assert my.mission_flags["sentinel-3a"] == 5 and my.mission_flags["sentinel-3b"] == 10
    assert cfg.require_verified("altimeter_l3_nrt").end is None


def test_verified_product_needs_a_dataset_id(raw_config):
    raw_config["products"]["insitu_buoys"]["dataset_id"] = None
    with pytest.raises(ConfigError, match="no dataset ID"):
        config_from_dict(raw_config)


def test_configured_missions_have_flag_codes(raw_config):
    cfg = config_from_dict(raw_config)
    missions = {cfg.splits.space.heldout_mission, *cfg.splits.space.train_missions}
    assert missions <= set(cfg.products["altimeter_l3_my"].mission_flags)
    raw_config["splits"]["space"]["train_missions"] = ["sentinel-3a", "unknown-sat"]
    with pytest.raises(ConfigError, match="mission_flags"):
        config_from_dict(raw_config)


def test_all_altimetry_from_one_product(raw_config):
    # The test year must not need the differently calibrated near-real-time altimetry.
    cfg = config_from_dict(raw_config)
    assert max(cfg.years) <= cfg.products["altimeter_l3_my"].end.year
    assert cfg.splits.test.start.year == 2023


def test_years_beyond_my_product_need_nrt_datasets(raw_config):
    raw_config["years"].append(2024)
    config_from_dict(raw_config)            # NRT has datasets for both configured missions
    raw_config["products"]["altimeter_l3_nrt"]["datasets"].pop("sentinel-3b")
    with pytest.raises(ConfigError, match="NRT"):
        config_from_dict(raw_config)


def test_wind_history_must_be_whole_steps(raw_config):
    raw_config["time"]["wind_history_hours"] = 48
    assert config_from_dict(raw_config).time.wind_history_hours == 48
    raw_config["time"]["wind_history_hours"] = 25
    with pytest.raises(ConfigError, match="wind_history_hours"):
        config_from_dict(raw_config)
