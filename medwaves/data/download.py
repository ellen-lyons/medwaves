"""Downloads for stage 1. Each one is subset to `domain`, the needed variables and dates, written
atomically under DATA_ROOT/raw/, and skipped if already there (a .json manifest beside each file
records the request; a different request for the same file is refused, not silently reused).

Credentials come only from environment variables (the Colab notebook sets them from Colab Secrets):
COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD for Copernicus Marine, CDSAPI_URL / CDSAPI_KEY for CDS.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from medwaves.config import Config
from medwaves.data.common import Layout, atomic_output, check_manifest, scratch_dir, write_manifest

BUOY_COLUMNS = ["platform_id", "platform_type", "time", "longitude", "latitude", "depth",
                "value", "value_qc"]
ALTIMETER_FILE = "ESACCI-SEASTATE-L3-SWH-MULTI_1D-{date}-fv01.nc"   # product manual, section 3
ALTIMETER_KEEP = ["lon", "lat", "swh_uncertainty", "distance_to_coast", "satellite"]


def _bbox(cfg: Config) -> dict:
    d = cfg.domain
    return {"lon_min": d.lon_min, "lon_max": d.lon_max, "lat_min": d.lat_min, "lat_max": d.lat_max}


def _http_get(url: str, out: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# Static files

def mesh(cfg: Config, layout: Layout) -> Path:
    path = layout.mesh_file
    request = {"url": cfg.mesh.url, "md5": cfg.mesh.md5}
    if check_manifest(path, request):
        return path
    with atomic_output(path) as tmp:
        _http_get(cfg.mesh.url, tmp)
        md5 = hashlib.md5(tmp.read_bytes()).hexdigest()
        if md5 != cfg.mesh.md5:
            raise RuntimeError(f"mesh md5 {md5} != config {cfg.mesh.md5}: the file on Zenodo changed")
    write_manifest(path, request)
    return path


def etopo(cfg: Config, layout: Layout) -> Path:
    path = layout.etopo
    product = cfg.require_verified("bathymetry")
    d = cfg.domain
    request = {"url": product.url, "variables": list(product.variables), **_bbox(cfg)}
    if check_manifest(path, request):
        return path
    var = product.variables[0]
    query = f"{var}[({d.lat_min}):1:({d.lat_max})][({d.lon_min}):1:({d.lon_max})]"
    with atomic_output(path) as tmp:
        _http_get(f"{product.url}?{urllib.parse.quote(query, safe='():')}", tmp)
        with xr.open_dataset(tmp) as ds:
            if var not in ds or ds[var].size == 0:
                raise RuntimeError(f"ETOPO response has no {var!r}: {list(ds)}")
    write_manifest(path, request)
    return path


# ERA5

def era5_month(cfg: Config, layout: Layout, year: int, month: int, days: list[int]) -> Path:
    path = layout.era5(year, month)
    product = cfg.require_verified("era5")
    d = cfg.domain
    request = {
        "product_type": ["reanalysis"],
        "variable": list(product.variables),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{x:02d}" for x in days],
        "time": [f"{h:02d}:00" for h in range(0, 24, cfg.time.step_hours)],
        "area": [d.lat_max, d.lon_min, d.lat_min, d.lon_max],     # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if check_manifest(path, request):
        return path
    import cdsapi

    with atomic_output(path) as tmp:
        cdsapi.Client(quiet=True, progress=False).retrieve(product.dataset, request, str(tmp))
        with xr.open_dataset(tmp) as ds:
            missing = {"u10", "v10"} - set(ds)
            if missing:
                raise RuntimeError(f"ERA5 file lacks {missing}: {list(ds)}")
    write_manifest(path, request)
    return path


# Buoys

def buoys_month(cfg: Config, layout: Layout, year: int, month: int, days: list[int]) -> Path:
    """Hourly buoy wave height for the given days of one month, in the toolbox's sparse format."""
    path = layout.buoys_raw(year, month)
    product = cfg.require_verified("insitu_buoys")
    start = dt.datetime(year, month, days[0])
    end = dt.datetime(year, month, days[-1]) + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    request = {"dataset_id": product.dataset_id, "variables": list(product.variables),
               "start": start.isoformat(), "end": end.isoformat(), **_bbox(cfg)}
    if check_manifest(path, request):
        return path
    import copernicusmarine

    d = cfg.domain
    tmpdir = scratch_dir() / "buoys"
    tmpdir.mkdir(parents=True, exist_ok=True)
    name = f"{year}-{month:02d}.parquet"
    if (tmpdir / name).exists():
        (tmpdir / name).unlink()
    copernicusmarine.subset(
        dataset_id=product.dataset_id, variables=list(product.variables),
        minimum_longitude=d.lon_min, maximum_longitude=d.lon_max,
        minimum_latitude=d.lat_min, maximum_latitude=d.lat_max,
        start_datetime=start, end_datetime=end,
        file_format="parquet", output_directory=tmpdir, output_filename=name,
        overwrite=True, disable_progress_bar=True,
    )
    # The toolbox writes nothing when no platform reported in the window.
    df = pd.read_parquet(tmpdir / name) if (tmpdir / name).exists() else pd.DataFrame(columns=BUOY_COLUMNS)
    missing = set(BUOY_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"buoy subset lacks columns {sorted(missing)}; got {list(df.columns)}")
    df = df[df["variable"].isin(product.variables)] if "variable" in df else df
    with atomic_output(path) as tmp:
        df[BUOY_COLUMNS].to_parquet(tmp, index=False)
    write_manifest(path, request, rows=len(df))
    (tmpdir / name).unlink(missing_ok=True)
    return path


# Altimetry (multi-year L3, merged daily global files)

def crop_altimeter_file(src: Path, cfg: Config, mission_codes: dict[str, int]) -> tuple[xr.Dataset, Counter]:
    """Crop one global daily file to `domain` and the given missions.

    Returns the cropped points (dimension `point`) and the in-domain point count per satellite
    code before the mission filter (for the report: which missions fly over the Med).
    """
    product = cfg.products["altimeter_l3_my"]
    d = cfg.domain
    with xr.open_dataset(src) as ds:
        lon, lat = ds["lon"].values, ds["lat"].values
        inside = (lon >= d.lon_min) & (lon <= d.lon_max) & (lat >= d.lat_min) & (lat <= d.lat_max)
        sat = ds["satellite"].values
        seen = Counter(int(c) for c in sat[inside])
        keep = inside & np.isin(sat, list(mission_codes.values()))
        idx = np.nonzero(keep)[0]
        out = ds[[product.swh_variable, *ALTIMETER_KEEP]].isel(time=idx).load()
    out = out.reset_coords().rename({"time": "point"})
    out["time"] = ("point", out["point"].values)
    out = out.drop_vars("point")
    return out, seen


def _get_altimeter_days(dataset_id: str, outdir: Path, year: int, month: int, days: list[int]) -> None:
    import copernicusmarine

    dates = "|".join(f"{year}{month:02d}{x:02d}" for x in days)
    copernicusmarine.get(
        dataset_id=dataset_id, regex=rf".*MULTI_1D-({dates})-fv01\.nc$",
        output_directory=outdir, no_directories=True, skip_existing=True, disable_progress_bar=True,
    )


def altimeter_month(cfg: Config, layout: Layout, year: int, month: int, days: list[int],
                    missions: list[str], fetch=_get_altimeter_days) -> Path:
    """Whole global daily files go to local scratch, are cropped, and only the crop is kept.

    `fetch(dataset_id, outdir, year, month, days)` puts the daily files in outdir (tests and
    --synthetic pass a fake one)."""
    path = layout.altimeter_raw(year, month)
    product = cfg.require_verified("altimeter_l3_my")
    codes = {m: product.mission_flags[m] for m in missions}
    request = {"dataset_id": product.dataset_id, "variables": [product.swh_variable, *ALTIMETER_KEEP],
               "year": year, "month": month, "days": days, "missions": codes, **_bbox(cfg)}
    if check_manifest(path, request):
        return path
    tmpdir = scratch_dir() / f"altimeter_{year}{month:02d}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    fetch(product.dataset_id, tmpdir, year, month, days)
    parts, seen, found = [], Counter(), []
    for x in days:
        f = tmpdir / ALTIMETER_FILE.format(date=f"{year}{month:02d}{x:02d}")
        if not f.exists():
            continue
        cropped, counts = crop_altimeter_file(f, cfg, codes)
        parts.append(cropped)
        seen.update(counts)
        found.append(x)
    data = xr.concat(parts, dim="point") if parts else xr.Dataset()
    with atomic_output(path) as tmp:
        data.to_netcdf(tmp)
    flag_names = {v: k for k, v in product.mission_flags.items()}
    write_manifest(path, request, days_found=found, days_missing=sorted(set(days) - set(found)),
                   domain_points_by_satellite={flag_names.get(c, f"code_{c}"): n
                                               for c, n in sorted(seen.items())})
    shutil.rmtree(tmpdir, ignore_errors=True)
    return path
