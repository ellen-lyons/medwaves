"""Synthetic raw data in the same formats the downloads produce, for laptop tests (no network).

The "sea" is a box (lon 2.5-9.5, lat 36.5-43.5) with an island at (6, 40); everything else is land.
Buoys include the cases QC must catch: bad flags, a spike, a stuck run, a drifting platform, a
co-located pair. Altimeter daily files mimic the merged multi-mission files (several missions,
points outside the domain) and go through the real crop step.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from medwaves.config import Config
from medwaves.data import download
from medwaves.data.common import Layout, months_in, stage_missions, stage_years, year_bounds

SEA = (2.5, 9.5, 36.5, 43.5)            # lon_min, lon_max, lat_min, lat_max
ISLAND = (6.0, 40.0, 0.4)               # lon, lat, radius in degrees
BUOYS = {                                # id: (lon, lat)
    "B_A": (3.5, 42.5), "B_A2": (3.55, 42.45),          # co-located pair (~7 km apart)
    "B_B": (8.5, 42.5), "B_C": (3.5, 37.5), "B_D": (8.5, 37.5),
    "B_E": (5.0, 41.0), "B_F": (7.5, 39.0), "B_G": (4.5, 39.5), "B_H": (7.0, 41.5),
}
DRIFTER = "D_X"


def write_mesh(path: Path, step: float = 0.5) -> None:
    lons = np.arange(SEA[0] + 0.5, SEA[1] - 0.49, step)
    lats = np.arange(SEA[2] + 0.5, SEA[3] - 0.49, step)
    nodes = [(x, y) for y in lats for x in lons]
    nx = len(lons)
    tris = []
    for i in range(len(lats) - 1):
        for j in range(nx - 1):
            a, b, c, d = i * nx + j, i * nx + j + 1, (i + 1) * nx + j, (i + 1) * nx + j + 1
            tris += [(a, b, d), (a, d, c)]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines += [f"{k + 1} {x} {y} 0" for k, (x, y) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(tris))]
    lines += [f"{k + 1} 2 2 1 1 {a + 1} {b + 1} {c + 1}" for k, (a, b, c) in enumerate(tris)]
    lines += ["$EndElements"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_etopo(path: Path, cfg: Config, res: float = 0.05) -> None:
    d = cfg.domain
    lat = np.arange(d.lat_min, d.lat_max + res / 2, res)
    lon = np.arange(d.lon_min, d.lon_max + res / 2, res)
    glon, glat = np.meshgrid(lon, lat)
    sea = (glon > SEA[0]) & (glon < SEA[1]) & (glat > SEA[2]) & (glat < SEA[3])
    sea &= np.hypot(glon - ISLAND[0], glat - ISLAND[1]) > ISLAND[2]
    alt = np.where(sea, -2000, 300).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"altitude": (("latitude", "longitude"), alt)},
               coords={"latitude": lat, "longitude": lon}).to_netcdf(path)


def buoy_frame(start: dt.datetime, end: dt.datetime, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="1h", inclusive="left")
    rows = []
    for k, (pid, (x, y)) in enumerate(BUOYS.items()):
        h = 1.2 + 0.8 * np.sin(np.arange(len(times)) / 9 + k) + rng.normal(0, 0.05, len(times))
        qc = np.ones(len(times), int)
        if pid == "B_B" and len(times) > 50:
            qc[5:8] = 4                                   # bad flags
            h[20] = 35.0                                  # spike (range)
            h[30:40] = 1.11                               # stuck run of 10 hours
            h[50] = np.nan                                # missing
        for t, v, q in zip(times, h.round(2), qc):
            rows.append((pid, "MO", t, x, y, 0.0, v, q))
        if pid == "B_C":                                  # a second depth: duplicate times
            rows += [(pid, "MO", t, x, y, 1.0, v, 1) for t, v in zip(times[:4], h[:4].round(2))]
    for n, t in enumerate(times):                          # drifting platform
        rows.append((DRIFTER, "DB", t, 4.0 + 0.02 * n, 38.0, 0.0, 1.0, 1))
    df = pd.DataFrame(rows, columns=["platform_id", "platform_type", "time", "longitude",
                                     "latitude", "depth", "value", "value_qc"])
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.insert(0, "variable", "VHM0")
    return df


def altimeter_day_file(path: Path, day: dt.date, codes: list[int], seed: int) -> None:
    """One fake merged daily file: two passes per mission over the sea box, plus far-away points."""
    rng = np.random.default_rng(seed)
    parts = []
    t0 = np.datetime64(dt.datetime(day.year, day.month, day.day, 2), "s")
    for m, code in enumerate(codes):
        for p in range(2):
            n = 160                                       # ~1 Hz, ~7 km apart, 8 degrees of latitude
            lat = np.linspace(SEA[2] - 0.3, SEA[3] + 0.3, n)
            lon = 3.0 + 1.3 * m + 0.6 * p + 0.15 * (lat - SEA[2])
            t = t0 + np.timedelta64(5 * 3600 * p + 1800 * m, "s") + np.arange(n).astype("timedelta64[s]")
            parts.append((t, lon, lat, np.full(n, code)))
        far = 20                                          # outside the domain: must be cropped away
        parts.append((t0 + np.arange(far).astype("timedelta64[s]"), np.full(far, 60.0),
                      np.linspace(0, 5, far), np.full(far, code)))
    t, lon, lat, sat = (np.concatenate(x) for x in zip(*parts))
    swh = 1.5 + 0.5 * np.sin(lat) + rng.normal(0, 0.1, lat.size)
    ds = xr.Dataset(
        {"swh": ("time", swh), "swh_adjusted": ("time", swh + 0.02), "swh_denoised": ("time", swh),
         "swh_uncertainty": ("time", np.full(lat.size, 0.1)),
         "distance_to_coast": ("time", np.full(lat.size, 50e3)),
         "bathymetry": ("time", np.full(lat.size, -2000.0)),
         "satellite": ("time", sat.astype(np.uint8))},
        coords={"time": t.astype("datetime64[ns]"), "lon": ("time", lon), "lat": ("time", lat)})
    ds.to_netcdf(path)


def era5_month_file(path: Path, cfg: Config, year: int, month: int, days: list[int]) -> None:
    d = cfg.domain
    lat = np.arange(d.lat_max, d.lat_min - 0.01, -0.25)       # descending, like CDS files
    lon = np.arange(d.lon_min, d.lon_max + 0.01, 0.25)
    times = [np.datetime64(dt.datetime(year, month, x, h)) for x in days
             for h in range(0, 24, cfg.time.step_hours)]
    shape = (len(times), lat.size, lon.size)
    u = np.broadcast_to(5 + 0.1 * lon[None, None, :], shape).astype(np.float32)
    v = np.broadcast_to(-2 + 0.1 * lat[None, :, None], shape).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), u),
                "v10": (("valid_time", "latitude", "longitude"), v)},
               coords={"valid_time": np.array(times, "datetime64[ns]"), "latitude": lat,
                       "longitude": lon}).to_netcdf(path)


def write_raw(cfg: Config, sample_days: int) -> None:
    """Fill DATA_ROOT/raw with synthetic files; existing files are left alone (like downloads)."""
    layout = Layout(cfg)
    if not layout.mesh_file.exists():
        write_mesh(layout.mesh_file)
    if not layout.etopo.exists():
        write_etopo(layout.etopo, cfg)
    for year in cfg.years:
        for y, m, days in months_in(*year_bounds(year, sample_days)):
            path = layout.buoys_raw(y, m)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                start = dt.datetime(y, m, days[0])
                buoy_frame(start, start + dt.timedelta(days=len(days)), cfg.seed + y).to_parquet(path)
    my = cfg.products["altimeter_l3_my"]
    codes = sorted(set(my.mission_flags.values()))       # every mission, so the crop has work to do
    for year in stage_years(cfg):
        for y, m, days in months_in(*year_bounds(year, sample_days)):
            if not layout.era5(y, m).exists():
                era5_month_file(layout.era5(y, m), cfg, y, m, days)

            def fake_get(dataset_id, outdir, yy, mm, dd):
                for x in dd:
                    altimeter_day_file(outdir / download.ALTIMETER_FILE.format(date=f"{yy}{mm:02d}{x:02d}"),
                                       dt.date(yy, mm, x), codes, seed=x)

            download.altimeter_month(cfg, layout, y, m, days, stage_missions(cfg), fetch=fake_get)
