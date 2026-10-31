"""Stage 1 driver. Steps run in order and each skips work already done, so a dead Colab session
just re-runs `medwaves-data`:

  download   mesh, ETOPO, buoys (all configured years: the held-out draw needs the test year),
             ERA5 and multi-year altimetry (stage years only)
  mesh       graph + node features -> processed/mesh.nc
  inventory  buoy QC for every year -> interim/buoys_qc/; inventory; held-out draft (once)
  process    altimeter QC + super-obs, buoy 3-hourly means, winds on nodes (stage years)
  stores     train_pool / heldout observation stores (stage years)
  report     data report (always rebuilt)
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import xarray as xr

from medwaves.config import Config
from medwaves.data import align, download, report
from medwaves.data.catalogue import years_available
from medwaves.data.common import (Layout, atomic_output, manifest_path, months_in, read_json,
                                  stage_missions, stage_years, write_json, year_bounds)
from medwaves.data.inventory import build_inventory, draw_heldout, inventory_hash
from medwaves.data.mesh import Bathymetry, Mesh, build_mesh_dataset
from medwaves.data.qc import qc_altimeter, qc_buoys
from medwaves.data.stores import stores_current, write_stores
from medwaves.splits import REPO_ROOT, SpaceSplit, draft_buoy_split_path, write_draft_buoy_split

STEPS = ("download", "mesh", "inventory", "process", "stores", "report")


def log(msg: str) -> None:
    print(f"[medwaves-data] {msg}", flush=True)


class Stage1:
    def __init__(self, cfg: Config, sample_days: int | None = None, live_catalogue: bool = True):
        if sample_days:
            # A few-day sample can never hold 30 test-year days: cap the eligibility threshold.
            space = dataclasses.replace(cfg.splits.space, heldout_min_test_days=min(
                cfg.splits.space.heldout_min_test_days, sample_days))
            cfg = dataclasses.replace(cfg, splits=dataclasses.replace(cfg.splits, space=space))
        self.cfg, self.sample_days, self.live = cfg, sample_days, live_catalogue
        self.layout = Layout(cfg)
        self._mesh = self._bathy = None

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            with xr.open_dataset(self.layout.mesh_nc) as ds:
                self._mesh = Mesh.from_dataset(ds.load())
        return self._mesh

    @property
    def bathy(self) -> Bathymetry:
        if self._bathy is None:
            self._bathy = Bathymetry(self.layout.etopo)
        return self._bathy

    def months(self, year: int):
        return months_in(*year_bounds(year, self.sample_days))

    def run(self, steps=STEPS) -> None:
        for step in steps:
            log(f"step {step}")
            getattr(self, f"step_{step}")()

    # Steps

    def step_download(self) -> None:
        cfg, lay = self.cfg, self.layout
        download.mesh(cfg, lay)
        download.etopo(cfg, lay)
        for year in cfg.years:
            for y, m, days in self.months(year):
                download.buoys_month(cfg, lay, y, m, days)
        for year in stage_years(cfg):
            for y, m, days in self.months(year):
                log(f"  ERA5 and altimetry {y}-{m:02d}")
                download.era5_month(cfg, lay, y, m, days)
                download.altimeter_month(cfg, lay, y, m, days, stage_missions(cfg))

    def step_mesh(self) -> None:
        lay = self.layout
        if lay.mesh_nc.exists():
            return
        m = self.cfg.mesh
        ds = build_mesh_dataset(lay.mesh_file, self.bathy, m.fetch_directions, m.fetch_max_km, m.fetch_step_km)
        with atomic_output(lay.mesh_nc) as tmp:
            ds.to_netcdf(tmp)
        log(f"  mesh: {ds.sizes['node']} nodes, {int(ds['on_land'].sum())} on land per ETOPO")

    def _qc_buoys_year(self, year: int) -> pd.DataFrame:
        path = self.layout.buoys_qc(year)
        if not path.exists():
            files = [self.layout.buoys_raw(y, m) for y, m, _ in self.months(year)]
            raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            res = qc_buoys(raw, self.cfg, self.mesh)
            write_json(path.with_suffix(".json"), {
                **res.summary(), "by_month": res.by_month.to_dict(orient="index"),
                "by_platform": res.by_platform.to_dict(orient="index"),
                "platform_types_raw": raw.groupby("platform_type")["platform_id"].nunique().to_dict()})
            with atomic_output(path) as tmp:
                res.kept.to_parquet(tmp, index=False)
        return pd.read_parquet(path)

    def step_inventory(self) -> None:
        cfg, lay = self.cfg, self.layout
        kept = {y: self._qc_buoys_year(y) for y in cfg.years}
        inv = build_inventory(kept, cfg)
        text = inv.to_csv(index=False)
        if not lay.inventory.exists() or lay.inventory.read_text() != text:
            with atomic_output(lay.inventory) as tmp:
                tmp.write_text(text)
        committed = REPO_ROOT / cfg.splits.space.heldout_buoys_file
        draft = draft_buoy_split_path(cfg)
        if committed.exists() or draft.exists():
            log(f"  buoy split already drawn ({'committed' if committed.exists() else 'draft'}); not redrawn")
            return
        heldout = draw_heldout(inv, cfg)
        path = write_draft_buoy_split(cfg, sorted(set(inv["platform_id"]) - heldout), sorted(heldout),
                                      seed=cfg.seed, inventory_hash=inventory_hash(inv))
        log(f"  drew {len(heldout)} held-out of {len(inv)} buoys -> {path}")

    def step_process(self) -> None:
        cfg, lay = self.cfg, self.layout
        for year in stage_years(cfg):
            buoys_out = lay.aligned("buoys", year)
            if not buoys_out.exists():
                aligned = align.align_buoys(self._qc_buoys_year(year), cfg, self.mesh)
                with atomic_output(buoys_out) as tmp:
                    aligned.to_netcdf(tmp)
            alt_out = lay.aligned("altimeter", year)
            if not alt_out.exists():
                files = [lay.altimeter_raw(y, m) for y, m, _ in self.months(year)]
                parts = [xr.open_dataset(f).load() for f in files]
                parts = [p for p in parts if p.sizes.get("point", 0)]
                raw = xr.concat(parts, dim="point") if parts else xr.Dataset()
                res = qc_altimeter(raw, cfg, self.mesh, self.bathy)
                superobs, stats = align.altimeter_superobs(res.kept, cfg, self.mesh)
                write_json(alt_out.with_suffix(".json"), {
                    **res.summary(), "by_month": res.by_month.to_dict(orient="index"), "superobs": stats})
                with atomic_output(alt_out) as tmp:
                    superobs.to_netcdf(tmp)
            wind_out = lay.wind(year)
            if not wind_out.exists():
                start, end = year_bounds(year, self.sample_days)
                wind = align.winds_to_nodes([lay.era5(y, m) for y, m, _ in self.months(year)],
                                            self.mesh, start, end, cfg.time.step_hours)
                with atomic_output(wind_out) as tmp:
                    wind.to_netcdf(tmp)

    def step_stores(self) -> None:
        cfg, lay = self.cfg, self.layout
        space = SpaceSplit.from_config(cfg)
        for year in stage_years(cfg):
            if stores_current(cfg, year, space):
                continue
            with xr.open_dataset(lay.aligned("buoys", year)) as b, \
                 xr.open_dataset(lay.aligned("altimeter", year)) as a:
                counts = write_stores(cfg, year, b.load(), a.load(), space)
            write_json(lay.interim / f"stores_{year}.json", counts)
            if not space.buoys_frozen:
                log("  buoys left out of the stores until the buoy split is committed (see report)")

    def step_report(self) -> None:
        cfg, lay = self.cfg, self.layout
        catalogue = years_available(cfg, live=self.live)
        space = SpaceSplit.from_config(cfg)
        committed = REPO_ROOT / cfg.splits.space.heldout_buoys_file
        draft = draft_buoy_split_path(cfg)
        if space.buoys_frozen:
            split = {"state": f"committed ({cfg.splits.space.heldout_buoys_file})",
                     "train_pool": len(space.train_buoys), "heldout": len(space.heldout_buoys),
                     "heldout_ids": sorted(space.heldout_buoys)}
        elif draft.exists():
            import yaml

            d = yaml.safe_load(draft.read_text())
            split = {"state": (f"DRAFT at {draft}, not active. Download it from Drive, copy it to "
                               f"{cfg.splits.space.heldout_buoys_file}, commit, push, then re-run "
                               "medwaves-data to put buoys into the stores."),
                     "train_pool": len(d["train_pool"]), "heldout": len(d["heldout"]),
                     "heldout_ids": d["heldout"]}
        else:
            split = {"state": "not drawn yet", "train_pool": None, "heldout": None}
        inv = pd.read_csv(lay.inventory, dtype={"platform_id": str}) if lay.inventory.exists() else pd.DataFrame()
        for year in stage_years(cfg):
            manifests = {}
            for y, m, _ in self.months(year):
                mp = manifest_path(lay.altimeter_raw(y, m))
                if mp.exists():
                    manifests[f"{y}-{m:02d}"] = read_json(mp).get("domain_points_by_satellite", {})
            alt_qc = read_json(lay.aligned("altimeter", year).with_suffix(".json"))
            stores_file = lay.interim / f"stores_{year}.json"
            with xr.open_dataset(lay.mesh_nc) as mesh, xr.open_dataset(lay.wind(year)) as wind, \
                 xr.open_dataset(lay.aligned("buoys", year)) as ab, \
                 xr.open_dataset(lay.aligned("altimeter", year)) as aa:
                text, _ = report.build(year, lay, {
                    "catalogue": catalogue, "mesh": mesh.load(), "wind": wind.load(),
                    "qc_buoys": read_json(lay.buoys_qc(year).with_suffix(".json")),
                    "qc_altimeter": alt_qc, "superobs": alt_qc.get("superobs", {}),
                    "aligned_buoys": ab.load(), "aligned_altimeter": aa.load(),
                    "altimeter_manifests": manifests, "inventory": inv, "split": split,
                    "stores": read_json(stores_file) if stores_file.exists() else {}})
            print(text)
            log(f"report written to {lay.report(year, 'md')}")


def probe(cfg: Config) -> None:
    """Download one day of each source to raw/probe/ and print what came back, before a full run."""
    pcfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_root=cfg.paths.raw / "probe"))
    lay = Layout(pcfg)
    year = cfg.pilot.year
    log(f"probe: one day ({year}-01-15) of each source into {pcfg.paths.data_root}")
    df = pd.read_parquet(download.buoys_month(pcfg, lay, year, 1, [15]))
    log(f"buoys: {len(df)} rows, columns {list(df.columns)}")
    if len(df):
        log(f"  platforms: {df['platform_id'].nunique()}; types: "
            f"{df.groupby('platform_type')['platform_id'].nunique().to_dict()}")
        log(f"  value_qc counts: {df['value_qc'].value_counts(dropna=False).to_dict()}")
        log(f"  depths: {sorted(df['depth'].dropna().unique().tolist())[:10]}")
    p = download.altimeter_month(pcfg, lay, year, 1, [15], stage_missions(cfg))
    with xr.open_dataset(p) as ds:
        log(f"altimeter crop: {ds.sizes.get('point', 0)} points, variables {list(ds.variables)}")
    log(f"  in-domain points by mission: {read_json(manifest_path(p))['domain_points_by_satellite']}")
    with xr.open_dataset(download.era5_month(pcfg, lay, year, 1, [15])) as ds:
        log(f"ERA5: dims {dict(ds.sizes)}, variables {list(ds.data_vars)}")
    log(f"catalogue: {years_available(cfg)}")
