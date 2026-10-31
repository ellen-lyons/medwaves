"""Buoy inventory over all configured years, and the held-out buoy draw.

Buoys closer than splits.space.buoy_cluster_km (single link) form a cluster, and whole clusters are
held out, so a held-out buoy never has a training buoy next to it. The draw is stratified by region
(coarse lon/lat boxes), uses only clusters with enough test-year data, and runs once: the result is
written as a draft (splits.write_draft_buoy_split) for the user to review and commit.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from medwaves.config import Config
from medwaves.data.common import km_to_chord, to_xyz

# Coarse regions for stratifying the draw: four longitude bands x north/south of 40N.
REGION_LON_EDGES = (5.0, 12.0, 20.0)
REGION_LAT_EDGE = 40.0


def region(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    band = np.searchsorted(REGION_LON_EDGES, lon, side="right")
    return np.array([f"lon{b}_{'N' if y >= REGION_LAT_EDGE else 'S'}" for b, y in zip(band, lat)])


def clusters(lon: np.ndarray, lat: np.ndarray, radius_km: float) -> np.ndarray:
    """Single-link cluster label per point (points closer than radius_km end up together)."""
    n = len(lon)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n and radius_km > 0:
        for i, j in cKDTree(to_xyz(lon, lat)).query_pairs(float(km_to_chord(radius_km))):
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    _, label = np.unique(roots, return_inverse=True)
    return label


def build_inventory(kept_by_year: dict[int, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """One row per buoy that passed QC in any configured year."""
    test = cfg.splits.test
    rows = []
    frames = [df.assign(year=y) for y, df in kept_by_year.items() if len(df)]
    if not frames:
        return pd.DataFrame(columns=["platform_id", "lon", "lat", "region", "cluster", "test_days"])
    df = pd.concat(frames, ignore_index=True)
    in_test = (df["time"] >= pd.Timestamp(test.start)) & (df["time"] < pd.Timestamp(test.end))
    test_days = df[in_test].groupby("platform_id")["time"].apply(lambda t: t.dt.normalize().nunique())
    for pid, g in df.groupby("platform_id", sort=True):
        row = {"platform_id": str(pid), "platform_type": g["platform_type"].iloc[0],
               "lon": float(g["lon"].median()), "lat": float(g["lat"].median()),
               "test_days": int(test_days.get(pid, 0))}
        for y in sorted(kept_by_year):
            row[f"hours_{y}"] = int((g["year"] == y).sum())
        rows.append(row)
    inv = pd.DataFrame(rows)
    inv["region"] = region(inv["lon"].values, inv["lat"].values)
    inv["cluster"] = clusters(inv["lon"].values, inv["lat"].values, cfg.splits.space.buoy_cluster_km)
    return inv


def inventory_hash(inv: pd.DataFrame) -> str:
    key = "\n".join(f"{p},{x:.3f},{y:.3f}" for p, x, y in
                    inv[["platform_id", "lon", "lat"]].sort_values("platform_id").itertuples(index=False))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def draw_heldout(inv: pd.DataFrame, cfg: Config) -> set[str]:
    """Held-out buoy IDs: whole clusters, per region, until each region is near the target fraction."""
    space = cfg.splits.space
    rng = np.random.default_rng(cfg.seed)
    clus = inv.groupby("cluster").agg(region=("region", "first"), size=("platform_id", "size"),
                                      test_days=("test_days", "max"))
    heldout_clusters = []
    for reg in sorted(inv["region"].unique()):
        target = space.heldout_buoy_fraction * int((inv["region"] == reg).sum())
        eligible = clus[(clus["region"] == reg) & (clus["test_days"] >= space.heldout_min_test_days)]
        held = 0
        for c in rng.permutation(eligible.index.values):
            size = int(eligible.loc[c, "size"])
            if abs(held + size - target) < abs(held - target):
                heldout_clusters.append(c)
                held += size
    # Small regions can round to zero: top up towards the overall target from any region.
    target = space.heldout_buoy_fraction * len(inv)
    held = int(clus.loc[heldout_clusters, "size"].sum())
    rest = clus[(clus["test_days"] >= space.heldout_min_test_days) & ~clus.index.isin(heldout_clusters)]
    for c in rng.permutation(rest.index.values):
        size = int(rest.loc[c, "size"])
        if abs(held + size - target) < abs(held - target):
            heldout_clusters.append(c)
            held += size
    return set(inv.loc[inv["cluster"].isin(heldout_clusters), "platform_id"])
