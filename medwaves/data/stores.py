"""Split aligned observations into the train_pool and heldout stores.

Every observation's role comes from SpaceSplit.role, which fails closed: an unknown buoy or mission
raises instead of landing in the train pool. Until the buoy split is committed to the repo, buoys
are left out of both stores (they wait in interim/) and only altimetry is written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from medwaves.config import Config
from medwaves.data.align import ALTIMETER, BUOY
from medwaves.data.common import atomic_output
from medwaves.splits import ObsRole, SpaceSplit


def store_path(cfg: Config, role: ObsRole, year: int) -> Path:
    """Write location (always data_root; readers go through splits.obs_store)."""
    return cfg.paths.processed / "obs" / ObsRole(role).value / f"{year}.nc"


def roles(obs: xr.Dataset, space: SpaceSplit) -> np.ndarray:
    cache: dict[tuple[str, str], str] = {}
    out = []
    for source, platform in zip(obs["source"].values, obs["platform"].values):
        key = (str(source), str(platform))
        if key not in cache:
            cache[key] = space.role(*key).value
        out.append(cache[key])
    return np.asarray(out, dtype=object)


def write_stores(cfg: Config, year: int, buoys: xr.Dataset | None, altimeter: xr.Dataset,
                 space: SpaceSplit) -> dict:
    """Write both stores for one year. Returns counts per store for the report."""
    parts = [altimeter] + ([buoys] if buoys is not None and space.buoys_frozen else [])
    obs = xr.concat(parts, dim="obs") if len(parts) > 1 else parts[0]
    obs = obs.isel(obs=np.argsort(obs["step_time"].values, kind="stable"))
    role = roles(obs, space)
    counts = {}
    for r in ObsRole:
        sel = obs.isel(obs=np.nonzero(role == r.value)[0])
        if r is ObsRole.TRAIN_POOL:
            space.assert_no_heldout(
                buoy_ids=sel["platform"].values[sel["source"].values == BUOY],
                missions=sel["platform"].values[sel["source"].values == ALTIMETER])
        sel.attrs.update(role=r.value, year=year, buoys_included=int(space.buoys_frozen))
        with atomic_output(store_path(cfg, r, year)) as tmp:
            sel.to_netcdf(tmp)
        counts[r.value] = {
            src: {"obs": int((sel["source"].values == src).sum()),
                  "platforms": sorted(set(sel["platform"].values[sel["source"].values == src].tolist()))}
            for src in (BUOY, ALTIMETER)}
    return counts


def stores_current(cfg: Config, year: int, space: SpaceSplit) -> bool:
    """True if both stores exist and were written with the buoy split in its current state."""
    for r in ObsRole:
        path = store_path(cfg, r, year)
        if not path.exists():
            return False
        with xr.open_dataset(path) as ds:
            if int(ds.attrs.get("buoys_included", -1)) != int(space.buoys_frozen):
                return False
    return True
