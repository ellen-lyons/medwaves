"""Shared helpers for stage 1: file layout, atomic writes, raw-file manifests, sphere geometry."""
from __future__ import annotations

import calendar
import contextlib
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from medwaves.config import Config

EARTH_RADIUS_KM = 6371.0
SCRATCH_ENV = "MEDWAVES_SCRATCH"   # local disk for whole global files before cropping (not Drive)


class Layout:
    """Every stage-1 file path, under data_root (writes) or the local processed copy (reads)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        p = cfg.paths
        self.raw, self.interim, self.processed, self.reports = p.raw, p.interim, p.processed, p.reports
        self.mesh_file = self.raw / "mesh" / Path(cfg.mesh.url.split("/files/")[1]).parts[0]
        self.etopo = self.raw / "etopo" / "etopo1_domain.nc"
        self.mesh_nc = self.processed / "mesh.nc"
        self.inventory = self.interim / "buoy_inventory.csv"

    def era5(self, year: int, month: int) -> Path:
        return self.raw / "era5" / str(year) / f"{year}-{month:02d}.nc"

    def buoys_raw(self, year: int, month: int) -> Path:
        return self.raw / "buoys" / str(year) / f"{year}-{month:02d}.parquet"

    def altimeter_raw(self, year: int, month: int) -> Path:
        return self.raw / "altimeter_my" / str(year) / f"{year}-{month:02d}.nc"

    def buoys_qc(self, year: int) -> Path:
        return self.interim / "buoys_qc" / f"{year}.parquet"

    def aligned(self, source: str, year: int) -> Path:
        return self.interim / "aligned" / f"{source}_{year}.nc"

    def wind(self, year: int) -> Path:
        return self.processed / "wind" / f"{year}.nc"

    def report(self, year: int, ext: str) -> Path:
        return self.reports / f"stage1_{year}.{ext}"


def scratch_dir() -> Path:
    import tempfile

    path = Path(os.environ.get(SCRATCH_ENV) or Path(tempfile.gettempdir()) / "medwaves_scratch")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def atomic_output(path: Path):
    """Yield a temporary path next to `path`; rename it into place only if the block succeeds.

    A Colab session that dies mid-write leaves a `.part` file, never a truncated output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path, obj) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def check_manifest(path: Path, request: dict) -> bool:
    """True if `path` exists and was made by the same request. Raises if made by a different one."""
    if not path.exists():
        return False
    mpath = manifest_path(path)
    if mpath.exists():
        saved = read_json(mpath).get("request")
        if saved != json.loads(json.dumps(request, sort_keys=True, default=str)):
            raise RuntimeError(f"{path} was downloaded with a different request:\n  saved {saved}\n"
                               f"  now   {request}\nDelete it (and its .json) to download again.")
    return True


def write_manifest(path: Path, request: dict, **extra) -> None:
    write_json(manifest_path(path), {"request": request, **extra})


# Time ranges

def year_bounds(year: int, sample_days: int | None) -> tuple[dt.datetime, dt.datetime]:
    """Half-open [start, end) of the year, or of its first `sample_days` days."""
    start = dt.datetime(year, 1, 1)
    end = dt.datetime(year + 1, 1, 1)
    if sample_days:
        end = min(end, start + dt.timedelta(days=sample_days))
    return start, end


def months_in(start: dt.datetime, end: dt.datetime) -> list[tuple[int, int, list[int]]]:
    """(year, month, days) for every month touched by [start, end)."""
    out = []
    t = dt.datetime(start.year, start.month, 1)
    while t < end:
        ndays = calendar.monthrange(t.year, t.month)[1]
        days = [d for d in range(1, ndays + 1) if start <= dt.datetime(t.year, t.month, d) < end]
        out.append((t.year, t.month, days))
        t = dt.datetime(t.year + (t.month == 12), t.month % 12 + 1, 1)
    return out


def stage_years(cfg: Config) -> list[int]:
    """Years stage 1 processes: the pilot year only, or every configured year."""
    return [cfg.pilot.year] if cfg.pilot.enabled else list(cfg.years)


def stage_missions(cfg: Config) -> list[str]:
    """Altimeter missions stage 1 keeps: train missions (pilot: just its one) plus the held-out one."""
    space = cfg.splits.space
    train = [cfg.pilot.mission] if cfg.pilot.enabled and cfg.pilot.mission else list(space.train_missions)
    return train + ([space.heldout_mission] if space.heldout_mission else [])


# Geometry on the sphere

def to_xyz(lon, lat) -> np.ndarray:
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def chord_to_km(chord):
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


def km_to_chord(km):
    return 2 * np.sin(np.asarray(km) / (2 * EARTH_RADIUS_KM))


def haversine_km(lon1, lat1, lon2, lat2):
    return chord_to_km(np.linalg.norm(to_xyz(lon1, lat1) - to_xyz(lon2, lat2), axis=-1))


def destination(lon, lat, bearing_deg, dist_km):
    """Point reached from (lon, lat) along a great circle with initial bearing, in degrees."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    b = np.radians(bearing_deg)
    d = np.asarray(dist_km) / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(b))
    lon2 = lon1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return (np.degrees(lon2) + 180) % 360 - 180, np.degrees(lat2)


def nearest_step(times, step_hours: int) -> np.ndarray:
    """Round datetime64 times to the nearest model step (ties go to the later step)."""
    t = np.asarray(times, "datetime64[s]").astype(np.int64)
    step = step_hours * 3600
    return ((t + step // 2) // step * step).astype("datetime64[s]")
