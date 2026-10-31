"""Alignment to the 3-hourly steps and the mesh: buoy step means, altimeter along-track
super-observations, links to the k nearest nodes, and ERA5 winds interpolated to the nodes."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from medwaves.config import Config
from medwaves.data.common import nearest_step
from medwaves.data.mesh import Mesh

BUOY, ALTIMETER = "buoy", "altimeter"


def _to_dataset(g: pd.DataFrame, source: str, mesh: Mesh, k: int) -> xr.Dataset:
    idx, dist = mesh.nearest(g["lon"].values, g["lat"].values, k) if len(g) else (
        np.zeros((0, k), np.int32), np.zeros((0, k), np.float32))
    return xr.Dataset(
        {
            "source": ("obs", np.full(len(g), source, dtype=object).astype(str)),
            "platform": ("obs", g["platform_id"].to_numpy(dtype=str)),
            "time": ("obs", g["time"].values.astype("datetime64[ns]")),
            "step_time": ("obs", g["step_time"].values.astype("datetime64[ns]")),
            "lon": ("obs", g["lon"].values.astype(np.float64)),
            "lat": ("obs", g["lat"].values.astype(np.float64)),
            "swh": ("obs", g["swh"].values.astype(np.float32)),
            "swh_std": ("obs", g["swh_std"].fillna(0).values.astype(np.float32)),
            "n_raw": ("obs", g["n_raw"].values.astype(np.int32)),
            "node_idx": (("obs", "k"), idx),
            "node_dist_km": (("obs", "k"), dist),
        },
        attrs={"note": "time is the mean observation time; step_time the model step it belongs to. "
                       "lon/lat are for evaluation maps, never model inputs."},
    )


def align_buoys(kept: pd.DataFrame, cfg: Config, mesh: Mesh) -> xr.Dataset:
    """Mean of each buoy's hourly values within +-step/2 of each step."""
    df = kept.assign(step_time=nearest_step(kept["time"].values, cfg.time.step_hours))
    g = (df.groupby(["platform_id", "step_time"], sort=True)
           .agg(time=("time", "mean"), lon=("lon", "median"), lat=("lat", "median"),
                swh=("swh", "mean"), swh_std=("swh", "std"), n_raw=("swh", "size"))
           .reset_index())
    return _to_dataset(g, BUOY, mesh, cfg.mesh.obs_k_nearest)


def altimeter_superobs(kept: pd.DataFrame, cfg: Config, mesh: Mesh) -> tuple[xr.Dataset, dict]:
    """Average 1 Hz points into one super-observation per pass and nearest mesh node.

    A pass is a run of one mission's points with no time gap over qc.altimeter_pass_gap_s. Super-obs
    from fewer than qc.altimeter_min_points points are dropped (counted in the returned stats).
    """
    q = cfg.qc
    if kept.empty:
        empty = kept.assign(step_time=kept["time"], swh_std=0.0, n_raw=0)
        return _to_dataset(empty, ALTIMETER, mesh, cfg.mesh.obs_k_nearest), {
            "superobs": 0, "superobs_dropped_few_points": 0, "points_in_dropped_superobs": 0}
    df = kept.sort_values(["platform_id", "time"], kind="stable").reset_index(drop=True)
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    new = np.ones(len(df), bool)
    new[1:] = (df["platform_id"].values[1:] != df["platform_id"].values[:-1]) | (np.diff(t) > q.altimeter_pass_gap_s)
    df["pass_id"] = np.cumsum(new)
    df["node"] = mesh.nearest(df["lon"].values, df["lat"].values, 1)[0][:, 0]
    g = (df.groupby(["platform_id", "pass_id", "node"], sort=True)
           .agg(time=("time", "mean"), lon=("lon", "mean"), lat=("lat", "mean"),
                swh=("swh", "mean"), swh_std=("swh", "std"), n_raw=("swh", "size"))
           .reset_index())
    few = g["n_raw"] < q.altimeter_min_points
    stats = {"superobs": int((~few).sum()), "superobs_dropped_few_points": int(few.sum()),
             "points_in_dropped_superobs": int(g.loc[few, "n_raw"].sum())}
    g = g[~few].sort_values("time", kind="stable")
    g["step_time"] = nearest_step(g["time"].values, cfg.time.step_hours)
    return _to_dataset(g, ALTIMETER, mesh, cfg.mesh.obs_k_nearest), stats


def winds_to_nodes(files: list[Path], mesh: Mesh, start: dt.datetime, end: dt.datetime,
                   step_hours: int) -> xr.Dataset:
    """Bilinear ERA5 u10/v10 at each mesh node, every step in [start, end)."""
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            parts.append(ds[["u10", "v10"]].load())
    ds = xr.concat(parts, dim="time").sortby("time").sortby("latitude").sortby("longitude")
    steps = pd.date_range(start, end, freq=f"{step_hours}h", inclusive="left")
    ds = ds.sel(time=ds["time"].isin(steps.values))
    missing = steps.difference(pd.DatetimeIndex(ds["time"].values))

    lat, lon = ds["latitude"].values, ds["longitude"].values
    fi = np.clip(np.interp(mesh.lat, lat, np.arange(lat.size)), 0, lat.size - 1.000001)
    fj = np.clip(np.interp(mesh.lon, lon, np.arange(lon.size)), 0, lon.size - 1.000001)
    i0, j0 = fi.astype(int), fj.astype(int)
    wi, wj = fi - i0, fj - j0
    out = {}
    for var in ("u10", "v10"):
        a = ds[var].values                                   # (time, lat, lon)
        out[var] = (("time", "node"), (
            a[:, i0, j0] * (1 - wi) * (1 - wj) + a[:, i0 + 1, j0] * wi * (1 - wj)
            + a[:, i0, j0 + 1] * (1 - wi) * wj + a[:, i0 + 1, j0 + 1] * wi * wj).astype(np.float32))
    res = xr.Dataset(out, coords={"time": ds["time"].values})
    res.attrs["missing_steps"] = len(missing)
    res.attrs["source"] = "ERA5 10 m wind, bilinear to mesh nodes"
    return res
