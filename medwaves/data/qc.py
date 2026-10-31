"""Quality control. Every rule tags the points it removes with a reason, so the report can show how
many points each rule drops (overall, per month, and per buoy for the stuck-value rule).

Rules run in order; a point is counted only under the first rule that removes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from medwaves.config import Config
from medwaves.data.common import haversine_km
from medwaves.data.mesh import Bathymetry, Mesh

BUOY_RULES = ("missing", "qc_flag", "outside_mesh", "drifting_platform", "position_outlier",
              "range", "duplicate", "stuck")
ALTIMETER_RULES = ("missing", "outside_mesh", "near_coast", "range")


@dataclass
class QCResult:
    kept: pd.DataFrame
    rules: tuple[str, ...]
    n_raw: int
    by_rule: dict[str, int]
    by_month: pd.DataFrame                    # rows: month, columns: raw, each rule, kept
    by_platform: pd.DataFrame = field(default_factory=pd.DataFrame)   # buoys: rows platform

    def summary(self) -> dict:
        return {"n_raw": self.n_raw, "n_kept": len(self.kept),
                "removed": {r: self.by_rule.get(r, 0) for r in self.rules}}


def _tally(df: pd.DataFrame, reason: np.ndarray, rules: tuple[str, ...], group: str) -> pd.DataFrame:
    month = df["time"].dt.strftime("%Y-%m")
    table = pd.crosstab(month if group == "month" else df[group], reason)
    table = table.reindex(columns=["", *rules], fill_value=0).rename(columns={"": "kept"})
    table.insert(0, "raw", table.sum(axis=1))
    return table[["raw", *rules, "kept"]]


def _result(df: pd.DataFrame, reason: np.ndarray, rules: tuple[str, ...], by_platform: bool) -> QCResult:
    counts = pd.Series(reason[reason != ""]).value_counts()
    return QCResult(
        kept=df[reason == ""].reset_index(drop=True),
        rules=rules,
        n_raw=len(df),
        by_rule={r: int(counts.get(r, 0)) for r in rules},
        by_month=_tally(df, reason, rules, "month") if len(df) else pd.DataFrame(),
        by_platform=_tally(df, reason, rules, "platform_id") if by_platform and len(df) else pd.DataFrame(),
    )


def _flag(reason: np.ndarray, name: str, mask) -> None:
    reason[(reason == "") & np.asarray(mask, bool)] = name


def stuck_runs(platform: np.ndarray, time: np.ndarray, value: np.ndarray, min_hours: int) -> np.ndarray:
    """True for points in runs of >= min_hours identical consecutive hourly values (input sorted).

    A run breaks when the platform changes, the value changes, or there is a gap over 1.5 h.
    """
    n = len(value)
    if n == 0:
        return np.zeros(0, bool)
    t = np.asarray(time, "datetime64[s]").astype(np.int64)
    new = np.ones(n, bool)
    new[1:] = (platform[1:] != platform[:-1]) | (value[1:] != value[:-1]) | (np.diff(t) > 5400)
    run = np.cumsum(new)
    length = np.bincount(run)[run]
    return length >= min_hours


def qc_buoys(raw: pd.DataFrame, cfg: Config, mesh: Mesh) -> QCResult:
    """raw: rows in the toolbox's sparse format (platform_id, time, longitude, latitude, value, value_qc...)."""
    q = cfg.qc
    df = pd.DataFrame({
        "platform_id": raw["platform_id"].astype(str).values,
        "platform_type": raw["platform_type"].astype(str).values,
        "time": pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None).values,
        "lon": raw["longitude"].astype(float).values,
        "lat": raw["latitude"].astype(float).values,
        "depth": pd.to_numeric(raw["depth"], errors="coerce").values,
        "swh": raw["value"].astype(float).values,
        "qc": pd.to_numeric(raw["value_qc"], errors="coerce").values,
    })
    df = df.sort_values(["platform_id", "time", "depth"], kind="stable").reset_index(drop=True)
    reason = np.full(len(df), "", dtype=object)

    _flag(reason, "missing", ~np.isfinite(df["swh"]) | ~np.isfinite(df["lon"]) | ~np.isfinite(df["lat"]))
    _flag(reason, "qc_flag", ~df["qc"].isin(q.buoy_good_qc_flags))
    ok = reason == ""
    inside = np.zeros(len(df), bool)
    inside[ok] = mesh.covers(df["lon"][ok], df["lat"][ok], q.obs_max_node_km)
    _flag(reason, "outside_mesh", ~inside)

    # Fixed buoys only: distance of each position from the platform's median position.
    ok = reason == ""
    med = df[ok].groupby("platform_id")[["lon", "lat"]].median()
    mlon = df["platform_id"].map(med["lon"]).values
    mlat = df["platform_id"].map(med["lat"]).values
    off = np.where(ok, haversine_km(df["lon"].values, df["lat"].values, mlon, mlat), 0.0)
    p95 = pd.Series(off[ok]).groupby(df["platform_id"][ok].values).quantile(0.95)
    drifting = df["platform_id"].isin(p95.index[p95 > q.buoy_max_drift_km]).values
    _flag(reason, "drifting_platform", drifting)
    _flag(reason, "position_outlier", off > q.buoy_max_drift_km)

    _flag(reason, "range", (df["swh"] < q.swh_min_m) | (df["swh"] > q.swh_max_m))
    ok = reason == ""
    dup = np.zeros(len(df), bool)
    dup[ok] = df[ok].duplicated(["platform_id", "time"], keep="first").values   # shallowest kept
    _flag(reason, "duplicate", dup)

    ok = np.nonzero(reason == "")[0]
    stuck = np.zeros(len(df), bool)
    stuck[ok] = stuck_runs(df["platform_id"].values[ok], df["time"].values[ok],
                           df["swh"].values[ok], q.buoy_stuck_hours)
    _flag(reason, "stuck", stuck)

    result = _result(df, reason, BUOY_RULES, by_platform=True)
    result.kept = result.kept.drop(columns=["qc", "depth"])
    return result


def qc_altimeter(raw: xr.Dataset, cfg: Config, mesh: Mesh, bathy: Bathymetry) -> QCResult:
    """raw: cropped multi-year L3 points (time, lon, lat, <swh_variable>, satellite, ...)."""
    product = cfg.products["altimeter_l3_my"]
    q = cfg.qc
    names = {code: m for m, code in product.mission_flags.items()}
    if raw.sizes.get("point", 0) == 0:
        empty = pd.DataFrame(columns=["platform_id", "time", "lon", "lat", "swh"])
        return QCResult(empty, ALTIMETER_RULES, 0, {}, pd.DataFrame())
    df = pd.DataFrame({
        "platform_id": [names[int(c)] for c in raw["satellite"].values],
        "time": pd.to_datetime(raw["time"].values),
        "lon": raw["lon"].values.astype(float),
        "lat": raw["lat"].values.astype(float),
        "swh": raw[product.swh_variable].values.astype(float),
    }).sort_values(["platform_id", "time"], kind="stable").reset_index(drop=True)
    reason = np.full(len(df), "", dtype=object)
    _flag(reason, "missing", ~np.isfinite(df["swh"]) | ~np.isfinite(df["lon"]) | ~np.isfinite(df["lat"]))
    ok = reason == ""
    inside = np.zeros(len(df), bool)
    inside[ok] = mesh.covers(df["lon"][ok], df["lat"][ok], q.obs_max_node_km)
    _flag(reason, "outside_mesh", ~inside)
    ok = reason == ""
    coast = np.full(len(df), np.inf)
    coast[ok] = bathy.distance_to_coast_km(df["lon"][ok], df["lat"][ok])
    _flag(reason, "near_coast", coast < q.altimeter_min_coast_km)
    _flag(reason, "range", (df["swh"] < q.swh_min_m) | (df["swh"] > q.swh_max_m))
    return _result(df, reason, ALTIMETER_RULES, by_platform=False)
