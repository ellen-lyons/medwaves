"""Years available per source, for the data report (decides later whether to extend before 2019)."""
from __future__ import annotations

from medwaves.config import Config

# From the multi-year altimeter product manual (CMEMS-WAVE-PUM-014-005, table 3): the merged files
# do not expose per-mission coverage in the catalogue.
ALTIMETER_MISSION_COVERAGE = {
    "cryosat-2": ("2010-07-16", "2023-12-31"),
    "saral": ("2013-03-14", "2023-12-31"),
    "jason-3": ("2016-02-17", "2023-12-31"),
    "sentinel-3a": ("2016-07-01", "2023-12-31"),
    "sentinel-3b": ("2018-05-08", "2023-12-31"),
    "sentinel-6a": ("2020-12-17", "2023-12-31"),
}


def _live_coverage(dataset_id: str) -> tuple[str, str] | None:
    """Time coverage of a Copernicus Marine dataset from the live catalogue (no login needed)."""
    import copernicusmarine

    cat = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    lo, hi = None, None
    for product in cat.products:
        for ds in product.datasets:
            if ds.dataset_id != dataset_id:
                continue
            for part in ds.versions[-1].parts:
                for service in part.services:
                    for var in service.variables:
                        for coord in var.coordinates:
                            if coord.coordinate_id != "time":
                                continue
                            if coord.minimum_value is None or coord.maximum_value is None:
                                continue
                            a, b = _date(coord.minimum_value), _date(coord.maximum_value)
                            lo = a if lo is None else min(lo, a)
                            hi = b if hi is None else max(hi, b)
    return None if lo is None else (lo, hi)


def _date(value) -> str:
    """Catalogue times are ISO strings or milliseconds since 1970 depending on the service."""
    import pandas as pd

    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="ms").date().isoformat()
    return pd.Timestamp(value).date().isoformat()


def years_available(cfg: Config, live: bool = True) -> dict[str, dict]:
    """Coverage per source: config dates (checked when recorded), plus the live catalogue if reachable."""
    out = {}
    for name, product in cfg.products.items():
        entry = {"start": str(product.start) if product.start else None,
                 "end": str(product.end) if product.end else ("present" if product.start else None),
                 "role": product.role}
        if live and product.dataset_id:
            try:
                entry["catalogue_now"] = _live_coverage(product.dataset_id)
            except Exception as e:                   # offline or catalogue change: report, don't fail
                entry["catalogue_now"] = f"unavailable ({type(e).__name__})"
        out[name] = entry
    out["altimeter_l3_my"]["missions"] = {m: {"start": a, "end": b}
                                          for m, (a, b) in ALTIMETER_MISSION_COVERAGE.items()}
    return out
