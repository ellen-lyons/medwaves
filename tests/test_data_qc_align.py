import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from medwaves.config import config_from_dict
from medwaves.data import synthetic
from medwaves.data.align import align_buoys, altimeter_superobs, winds_to_nodes
from medwaves.data.common import nearest_step
from medwaves.data.inventory import build_inventory, clusters, draw_heldout
from medwaves.data.mesh import Bathymetry, Mesh, read_gmsh
from medwaves.data.qc import BUOY_RULES, qc_altimeter, qc_buoys, stuck_runs

START = dt.datetime(2019, 1, 1)


@pytest.fixture
def env(raw_config, tmp_path):
    cfg = config_from_dict(raw_config, tmp_path)
    synthetic.write_mesh(tmp_path / "m.msh")
    synthetic.write_etopo(tmp_path / "e.nc", cfg)
    return cfg, Mesh(*read_gmsh(tmp_path / "m.msh")), Bathymetry(tmp_path / "e.nc")


# Buoy QC

def test_stuck_runs():
    p = np.array(["a"] * 8 + ["b"] * 3)
    t = np.array([START + dt.timedelta(hours=h) for h in [0, 1, 2, 3, 4, 5, 6, 9, 0, 1, 2]], "datetime64[s]")
    v = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2.0])
    # a: six identical consecutive hours (1-6); hour 9 follows a gap. b starts a new run.
    assert stuck_runs(p, t, v, 6).tolist() == [False] + [True] * 6 + [False] * 4
    assert not stuck_runs(p, t, v, 7).any()


def test_buoy_qc_rules(env):
    cfg, mesh, _ = env
    raw = synthetic.buoy_frame(START, START + dt.timedelta(days=3), seed=1)
    res = qc_buoys(raw, cfg, mesh)
    r = res.by_rule
    assert r["missing"] == 1 and r["qc_flag"] == 3 and r["range"] == 1
    assert r["duplicate"] == 4                                    # second depth at B_C
    assert r["stuck"] == 10                                       # the 10-hour run at B_B
    assert r["drifting_platform"] == 72 and synthetic.DRIFTER not in set(res.kept["platform_id"])
    assert res.n_raw == len(raw) == len(res.kept) + sum(r.values())
    month = res.by_month.loc["2019-01"]
    assert month["raw"] == len(raw) and month["kept"] == len(res.kept)
    assert res.by_platform.loc["B_B", "stuck"] == 10
    assert list(res.by_month.columns) == ["raw", *BUOY_RULES, "kept"]


def test_buoy_qc_flags_from_config(env):
    cfg, mesh, _ = env
    raw = synthetic.buoy_frame(START, START + dt.timedelta(days=1), seed=1)
    raw.loc[raw["platform_id"] == "B_D", "value_qc"] = 2          # "probably good"
    assert "B_D" in set(qc_buoys(raw, cfg, mesh).kept["platform_id"])
    strict = cfg.__class__(**{**cfg.__dict__, "qc": cfg.qc.__class__(
        **{**cfg.qc.__dict__, "buoy_good_qc_flags": (1,)})})
    assert "B_D" not in set(qc_buoys(raw, strict, mesh).kept["platform_id"])


def _alt_points(cfg, lon, lat, sat="sentinel-3a", swh=1.5, t0=START):
    n = len(lon)
    return xr.Dataset({
        "time": ("point", np.array([t0 + dt.timedelta(seconds=i) for i in range(n)], "datetime64[ns]")),
        "lon": ("point", np.asarray(lon, float)), "lat": ("point", np.asarray(lat, float)),
        "swh_adjusted": ("point", np.full(n, swh)),
        "satellite": ("point", np.full(n, cfg.products["altimeter_l3_my"].mission_flags[sat], np.uint8))})


def test_altimeter_qc_drops_near_coast_and_outside(env):
    cfg, mesh, bathy = env
    # Open sea, 10 km off the island shore, outside the mesh, NaN, too high.
    shore_lon = synthetic.ISLAND[0] - synthetic.ISLAND[2] - 10 / (111.2 * np.cos(np.radians(40)))
    raw = _alt_points(cfg, [4.0, shore_lon, 0.0, 4.0, 4.1], [40.0, 40.0, 40.0, 41.0, 41.0])
    raw["swh_adjusted"].values[3] = np.nan
    raw["swh_adjusted"].values[4] = 25.0
    res = qc_altimeter(raw, cfg, mesh, bathy)
    assert res.by_rule == {"missing": 1, "outside_mesh": 1, "near_coast": 1, "range": 1}
    assert len(res.kept) == 1 and res.kept["platform_id"].iloc[0] == "sentinel-3a"


# Alignment

def test_nearest_step():
    t = np.array(["2019-01-01T01:29", "2019-01-01T01:30", "2019-01-01T04:00"], "datetime64[s]")
    assert nearest_step(t, 3).astype(str).tolist() == [
        "2019-01-01T00:00:00", "2019-01-01T03:00:00", "2019-01-01T03:00:00"]


def test_align_buoys(env):
    cfg, mesh, _ = env
    kept = pd.DataFrame({"platform_id": "B1", "platform_type": "MO",
                         "time": pd.date_range(START, periods=6, freq="1h"),
                         "lon": 5.0, "lat": 40.5, "swh": [1, 2, 3, 4, 5, 6.0]})
    ds = align_buoys(kept, cfg, mesh)
    # Hours 00, 01 -> step 00; 02, 03, 04 -> step 03; 05 -> step 06.
    assert ds["swh"].values.tolist() == [1.5, 4.0, 6.0]
    assert ds["n_raw"].values.tolist() == [2, 3, 1]
    assert ds["node_idx"].shape == (3, cfg.mesh.obs_k_nearest)
    d = ds["node_dist_km"].values
    assert (np.diff(d, axis=1) >= 0).all()


def test_altimeter_superobs_group_by_pass_and_node(env):
    cfg, mesh, _ = env
    kept = pd.DataFrame({
        "platform_id": ["sentinel-3a"] * 5 + ["sentinel-3a"] * 2,
        "time": [START + dt.timedelta(seconds=s) for s in [0, 1, 2, 3, 4, 600, 601]],
        "lon": [5.0] * 7, "lat": [40.5] * 7, "swh": [1, 1, 1, 1, 1, 2, 2.0]})
    ds, stats = altimeter_superobs(kept, cfg, mesh)
    # Second group is a new pass (600 s gap) with only 2 points: below the minimum of 3.
    assert ds.sizes["obs"] == 1 and float(ds["swh"][0]) == 1.0 and int(ds["n_raw"][0]) == 5
    assert stats == {"superobs": 1, "superobs_dropped_few_points": 1, "points_in_dropped_superobs": 2}


def test_winds_bilinear_exact_for_linear_field(env, tmp_path):
    cfg, mesh, _ = env
    synthetic.era5_month_file(tmp_path / "w.nc", cfg, 2019, 1, [1, 2])
    w = winds_to_nodes([tmp_path / "w.nc"], mesh, START, START + dt.timedelta(days=2), 3)
    assert w.sizes == {"time": 16, "node": mesh.lon.size}
    np.testing.assert_allclose(w["u10"].values[0], 5 + 0.1 * mesh.lon, atol=1e-4)
    np.testing.assert_allclose(w["v10"].values[0], -2 + 0.1 * mesh.lat, atol=1e-4)
    assert w.attrs["missing_steps"] == 0


# Inventory and held-out draw

def test_clusters_single_link():
    # a-b 20 km, b-c 20 km, a-c 40 km: one chain at 30 km; d far away.
    lon = np.array([5.0, 5.0, 5.0, 9.0])
    lat = np.array([40.0, 40.18, 40.36, 40.0])
    lab = clusters(lon, lat, 30)
    assert lab[0] == lab[1] == lab[2] != lab[3]
    assert len(set(clusters(lon, lat, 10))) == 4


def _inventory(cfg, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x, y = rng.uniform(-5, 35), rng.uniform(31, 45)
        times = pd.date_range("2023-01-01" if i % 4 else "2021-01-01", periods=24 * 40, freq="1h")
        rows.append(pd.DataFrame({"platform_id": f"P{i:02d}", "platform_type": "MO", "time": times,
                                  "lon": x, "lat": y, "swh": 1.0}))
        if i % 5 == 0:                                            # a near-twin 5 km away
            rows.append(rows[-1].assign(platform_id=f"P{i:02d}b", lon=x + 0.05))
    df = pd.concat(rows)
    return build_inventory({y: df[df["time"].dt.year == y] for y in (2021, 2023)}, cfg)


def test_heldout_draw_keeps_clusters_whole(raw_config):
    cfg = config_from_dict(raw_config)
    inv = _inventory(cfg)
    held = draw_heldout(inv, cfg)
    by_cluster = inv.assign(h=inv["platform_id"].isin(held)).groupby("cluster")["h"]
    assert (by_cluster.nunique() == 1).all()                      # never split a cluster
    assert abs(len(held) / len(inv) - cfg.splits.space.heldout_buoy_fraction) < 0.08
    # Only buoys with enough test-year data.
    assert (inv.set_index("platform_id").loc[sorted(held), "test_days"] >= 30).all()
    assert draw_heldout(inv, cfg) == held                         # deterministic with the seed
    assert "P00b" in set(inv["platform_id"]) and len({c for c in inv["cluster"]}) < len(inv)
