"""End-to-end stage 1 on synthetic data (no network), including the leakage checks on the stores."""
import shutil

import numpy as np
import pytest
import xarray as xr
import yaml

from medwaves.config import config_from_dict
from medwaves.data import download, synthetic
from medwaves.data.align import ALTIMETER, BUOY
from medwaves.data.common import Layout, check_manifest, read_json, manifest_path
from medwaves.data.pipeline import Stage1
from medwaves.data.stores import store_path
from medwaves.splits import LeakageError, ObsRole, SpaceSplit, obs_store, INPUT, TARGET

DAYS = 3


@pytest.fixture
def run(raw_config, tmp_path):
    committed = tmp_path / "repo" / "heldout_buoys.yaml"           # stands in for configs/
    raw_config["splits"]["space"]["heldout_buoys_file"] = str(committed)
    cfg = config_from_dict(raw_config, tmp_path / "data")
    synthetic.write_raw(cfg, DAYS)
    stage = Stage1(cfg, DAYS, live_catalogue=False)
    stage.run()
    return cfg, stage, committed


def _commit(cfg, committed):
    committed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.paths.draft_splits / committed.name, committed)


def test_crop_keeps_only_configured_missions_in_domain(run):
    cfg, stage, _ = run
    path = stage.layout.altimeter_raw(2019, 1)
    with xr.open_dataset(path) as ds:
        codes = set(np.unique(ds["satellite"].values).tolist())
        assert codes == {5, 10}                                      # sentinel-3a, sentinel-3b
        assert (ds["lon"] <= cfg.domain.lon_max).all()
    seen = read_json(manifest_path(path))["domain_points_by_satellite"]
    assert set(seen) == set(cfg.products["altimeter_l3_my"].mission_flags)   # all missions reported


def test_draft_split_then_buoys_wait_until_committed(run):
    cfg, stage, committed = run
    draft = yaml.safe_load((cfg.paths.draft_splits / committed.name).read_text())
    assert draft["heldout"] and not set(draft["heldout"]) & set(draft["train_pool"])
    for role in ObsRole:
        with xr.open_dataset(store_path(cfg, role, 2019)) as ds:
            assert BUOY not in set(ds["source"].values)               # no buoys before the commit
            assert ds.attrs["buoys_included"] == 0
    assert (stage.layout.reports / "stage1_2019.md").read_text().count("DRAFT") == 1


def test_stores_never_leak_heldout_into_train_pool(run):
    cfg, stage, committed = run
    _commit(cfg, committed)
    stage.run(["stores", "report"])
    space = SpaceSplit.from_config(cfg)
    assert space.buoys_frozen
    with xr.open_dataset(store_path(cfg, ObsRole.TRAIN_POOL, 2019)) as tp, \
         xr.open_dataset(store_path(cfg, ObsRole.HELDOUT, 2019)) as ho:
        assert tp.attrs["buoys_included"] == 1
        tp_buoys = set(tp["platform"].values[tp["source"].values == BUOY])
        ho_buoys = set(ho["platform"].values[ho["source"].values == BUOY])
        assert tp_buoys and ho_buoys
        assert not tp_buoys & space.heldout_buoys and tp_buoys <= space.train_buoys
        assert ho_buoys <= space.heldout_buoys
        assert set(tp["platform"].values[tp["source"].values == ALTIMETER]) == {"sentinel-3a"}
        assert set(ho["platform"].values[ho["source"].values == ALTIMETER]) == {"sentinel-3b"}
        space.assert_no_heldout(tp_buoys, set(tp["platform"].values[tp["source"].values == ALTIMETER]))
    # The heldout store can be found as a target, never as an input.
    assert obs_store(cfg.paths, ObsRole.HELDOUT, TARGET).parent.exists()
    with pytest.raises(LeakageError):
        obs_store(cfg.paths, ObsRole.HELDOUT, INPUT)


def test_unknown_buoy_fails_closed(run):
    cfg, stage, committed = run
    _commit(cfg, committed)
    split = yaml.safe_load(committed.read_text())
    split["train_pool"].remove("B_D")
    committed.write_text(yaml.safe_dump(split))
    with pytest.raises(LeakageError, match="B_D"):
        stage.run(["stores"])


def test_rerun_does_no_work(run):
    cfg, stage, _ = run
    outputs = [p for p in cfg.paths.data_root.rglob("*") if p.is_file() and "reports" not in p.parts]
    before = {p: p.stat().st_mtime_ns for p in outputs}
    stage.run()
    assert {p: p.stat().st_mtime_ns for p in outputs} == before
    assert not list(cfg.paths.data_root.rglob("*.part"))


def test_changed_request_is_refused(run):
    cfg, stage, _ = run
    path = stage.layout.altimeter_raw(2019, 1)
    request = read_json(manifest_path(path))["request"]
    assert check_manifest(path, request)
    with pytest.raises(RuntimeError, match="different request"):
        check_manifest(path, {**request, "missions": {"jason-3": 3}})


def test_report_contents(run):
    cfg, stage, _ = run
    report = read_json(stage.layout.report(2019, "json"))
    assert report["qc"]["buoys"]["removed"]["stuck"] == 10            # the synthetic stuck run
    assert set(report["monthly"]["2019-01"]) >= {"buoy raw (hourly)", "buoy after QC", "ERA5 steps"}
    assert report["monthly"]["2019-01"]["ERA5 steps"] == DAYS * 8
    assert report["split"]["heldout"] >= 1
    assert "altimeter_l3_my" in report["catalogue"]
    text = stage.layout.report(2019, "md").read_text()
    assert "Stuck-value rule" in text and "Years available per source" in text


def test_wind_and_mesh_outputs(run):
    cfg, stage, _ = run
    with xr.open_dataset(stage.layout.wind(2019)) as w, xr.open_dataset(stage.layout.mesh_nc) as m:
        assert w.sizes == {"time": DAYS * 8, "node": m.sizes["node"]}
        assert not np.isnan(w["u10"].values).any()
        assert m["fetch_km"].shape[1] == cfg.mesh.fetch_directions


def test_download_skips_existing(run, monkeypatch):
    cfg, stage, _ = run

    def boom(*a, **k):
        raise AssertionError("tried to download")

    monkeypatch.setattr(download, "_http_get", boom)
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", boom)
    monkeypatch.setattr(copernicusmarine, "get", boom)
    stage.step_download()                                        # everything is already there
    assert Layout(cfg).mesh_file.exists()
