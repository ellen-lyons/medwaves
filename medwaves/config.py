"""Load and validate the single YAML config into frozen dataclasses."""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
import random
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DATA_ROOT_ENV = "MEDWAVES_DATA_ROOT"
LOCAL_ROOT_ENV = "MEDWAVES_LOCAL_ROOT"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Paths:
    """data_root (Drive on Colab) is the durable store that every stage writes to.

    local_root, if set, is a fast local copy of processed/ (made by the notebook at session
    start) that stages read from instead of Drive.
    """
    data_root: Path
    local_root: Path | None = None

    @property
    def processed_read(self) -> Path:
        """Where to read processed data: the local copy if there is one, else data_root."""
        return (self.local_root or self.data_root) / "processed"

    @property
    def draft_splits(self) -> Path:
        """Where stage 1 writes the drawn buoy split for the user to copy into the repo."""
        return self.data_root / "splits"

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def checkpoints(self) -> Path:
        return self.data_root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.data_root / "reports"


@dataclass(frozen=True)
class Pilot:
    enabled: bool
    year: int
    mission: str | None
    local_sample_days: int


@dataclass(frozen=True)
class Time:
    step_hours: int
    wind_history_hours: int
    obs_history_hours: int


@dataclass(frozen=True)
class Interval:
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class Space:
    heldout_buoy_fraction: float
    buoy_cluster_km: float
    heldout_min_test_days: int
    heldout_buoys_file: str
    heldout_mission: str | None
    train_missions: tuple[str, ...]


@dataclass(frozen=True)
class Splits:
    gap_days: int
    train: Interval
    val: Interval
    test: Interval
    space: Space
    masking_fraction: float


@dataclass(frozen=True)
class QC:
    altimeter_min_coast_km: float
    altimeter_good_flags: tuple[int, ...] | None
    altimeter_min_points: int
    altimeter_pass_gap_s: float
    swh_min_m: float
    swh_max_m: float
    obs_max_node_km: float
    buoy_good_qc_flags: tuple[int, ...]
    buoy_max_drift_km: float
    buoy_stuck_hours: int


@dataclass(frozen=True)
class Mesh:
    source: str
    url: str
    md5: str
    resolution_km: float
    obs_k_nearest: int
    fetch_directions: int
    fetch_max_km: float
    fetch_step_km: float


@dataclass(frozen=True)
class Features:
    directional_fetch: bool


@dataclass(frozen=True)
class Domain:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class Product:
    verified: bool = False
    product_id: str | None = None
    dataset_id: str | None = None
    datasets: dict[str, str] = field(default_factory=dict)   # mission label -> dataset ID
    dataset: str | None = None                               # CDS / ERDDAP dataset name
    url: str | None = None
    variables: tuple[str, ...] = ()
    swh_variable: str | None = None
    mission_flags: dict[str, int] = field(default_factory=dict)  # mission label -> `satellite` code
    start: dt.date | None = None                             # catalogue time coverage
    end: dt.date | None = None                               # null: still running
    role: str = "observation"


@dataclass(frozen=True)
class Model:
    latent: int
    message_passing_steps: int


@dataclass(frozen=True)
class Train:
    batch_size: int | None
    lr: float
    epochs: int | None
    checkpoint_every_epoch: bool


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    years: tuple[int, ...]
    pilot: Pilot
    domain: Domain
    time: Time
    splits: Splits
    qc: QC
    mesh: Mesh
    features: Features
    products: dict[str, Product]
    model: Model
    train: Train

    def require_verified(self, name: str) -> Product:
        """Return a product, refusing if its IDs have not been looked up and verified."""
        if name not in self.products:
            raise ConfigError(f"unknown product {name!r}")
        product = self.products[name]
        if not product.verified:
            raise ConfigError(
                f"product {name!r} is not verified: look up its IDs, record them in the "
                "config and set verified: true"
            )
        return product


def _build(tp, value, where: str):
    """Build `tp` from parsed YAML, rejecting unknown keys and wrong types."""
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        hints = typing.get_type_hints(tp)
        unknown = set(value) - {f.name for f in dataclasses.fields(tp)}
        if unknown:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
        kwargs = {k: _build(hints[k], v, f"{where}.{k}") for k, v in value.items()}
        try:
            return tp(**kwargs)
        except TypeError as e:
            raise ConfigError(f"{where}: {e}") from None
    if origin in (typing.Union, types.UnionType):
        if value is None:
            if type(None) not in typing.get_args(tp):
                raise ConfigError(f"{where}: must not be null")
            return None
        (inner,) = [a for a in typing.get_args(tp) if a is not type(None)]
        return _build(inner, value, where)
    if value is None:
        raise ConfigError(f"{where}: must not be null")
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        _, value_tp = typing.get_args(tp)
        return {k: _build(value_tp, v, f"{where}.{k}") for k, v in value.items()}
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        item_tp = typing.get_args(tp)[0]
        return tuple(_build(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))
    if tp is Path:
        return Path(value)
    if tp is dt.date:
        if isinstance(value, dt.date):
            return value
        try:
            return dt.date.fromisoformat(str(value))
        except ValueError:
            raise ConfigError(f"{where}: expected a YYYY-MM-DD date") from None
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if (tp is int and isinstance(value, bool)) or not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _validate(cfg: Config) -> None:
    s = cfg.splits
    intervals = [("train", s.train), ("val", s.val), ("test", s.test)]
    for name, iv in intervals:
        if iv.start >= iv.end:
            raise ConfigError(f"splits.{name}: start must be before end")
    for (a, ia), (b, ib) in zip(intervals, intervals[1:]):
        if ib.start < ia.end:
            raise ConfigError(f"splits.{b} overlaps or precedes splits.{a}")
    if s.gap_days < 0:
        raise ConfigError("splits.gap_days must be >= 0")
    for name, iv in intervals:
        if iv.start.year < min(cfg.years) or (iv.end - dt.timedelta(days=1)).year > max(cfg.years):
            raise ConfigError(f"splits.{name} falls outside years {list(cfg.years)}")
    for key, value in [
        ("splits.space.heldout_buoy_fraction", s.space.heldout_buoy_fraction),
        ("splits.masking_fraction", s.masking_fraction),
    ]:
        if not 0.0 < value < 1.0:
            raise ConfigError(f"{key} must be between 0 and 1")
    if s.space.heldout_mission is not None and s.space.heldout_mission in s.space.train_missions:
        raise ConfigError("splits.space.heldout_mission is also listed in train_missions")
    missions = {*s.space.train_missions, *([s.space.heldout_mission] if s.space.heldout_mission else [])}
    my = cfg.products.get("altimeter_l3_my")
    if my is not None:
        missing = missions - set(my.mission_flags)
        if missing:
            raise ConfigError(f"missions {sorted(missing)} have no products.altimeter_l3_my.mission_flags code")
        if my.end and max(cfg.years) > my.end.year:
            # Later years would need the near-real-time product, calibrated differently.
            nrt = cfg.products.get("altimeter_l3_nrt")
            if nrt is None or missions - set(nrt.datasets):
                raise ConfigError("years after the multi-year altimeter product need NRT datasets "
                                  "for every configured mission")
    if s.space.buoy_cluster_km < 0 or s.space.heldout_min_test_days < 0:
        raise ConfigError("splits.space.buoy_cluster_km and heldout_min_test_days must be >= 0")
    if cfg.pilot.mission is not None:
        if cfg.pilot.mission == s.space.heldout_mission:
            raise ConfigError("pilot.mission must not be the held-out mission")
        if cfg.pilot.mission not in s.space.train_missions:
            raise ConfigError("pilot.mission must be one of splits.space.train_missions")
    reanalysis = cfg.products.get("wave_reanalysis")
    if reanalysis is not None and reanalysis.role != "referee_only":
        raise ConfigError("products.wave_reanalysis.role must be referee_only (never an input or target)")
    for name, product in cfg.products.items():
        if name != "wave_reanalysis" and product.role != "observation":
            raise ConfigError(f"products.{name}.role must be observation")
        if product.verified and not (product.dataset_id or product.datasets or product.dataset):
            raise ConfigError(f"products.{name} is marked verified but has no dataset ID")
        if product.start and product.end and product.start > product.end:
            raise ConfigError(f"products.{name}: start is after end")
    d = cfg.domain
    if not (d.lon_min < d.lon_max and d.lat_min < d.lat_max):
        raise ConfigError("domain: min must be below max")
    if cfg.pilot.year not in cfg.years:
        raise ConfigError("pilot.year must be one of years")
    if cfg.time.step_hours <= 0 or 24 % cfg.time.step_hours:
        raise ConfigError("time.step_hours must divide 24")
    for key in ("wind_history_hours", "obs_history_hours"):
        if getattr(cfg.time, key) % cfg.time.step_hours:
            raise ConfigError(f"time.{key} must be a multiple of time.step_hours")
    if cfg.mesh.source != "wavegraph":
        raise ConfigError("mesh.source: only 'wavegraph' (the released Zenodo mesh) is supported")
    if not cfg.qc.swh_min_m < cfg.qc.swh_max_m:
        raise ConfigError("qc.swh_min_m must be below qc.swh_max_m")


def config_from_dict(raw: dict, data_root: str | os.PathLike | None = None) -> Config:
    """Build a validated Config.

    data_root priority: argument > $MEDWAVES_DATA_ROOT > YAML. local_root: $MEDWAVES_LOCAL_ROOT > YAML.
    """
    cfg = _build(Config, raw, "config")
    paths = cfg.paths
    override = data_root or os.environ.get(DATA_ROOT_ENV)
    if override:
        paths = dataclasses.replace(paths, data_root=Path(override).expanduser())
    if os.environ.get(LOCAL_ROOT_ENV):
        paths = dataclasses.replace(paths, local_root=Path(os.environ[LOCAL_ROOT_ENV]).expanduser())
    cfg = dataclasses.replace(cfg, paths=paths)
    _validate(cfg)
    return cfg


def load_config(path: str | os.PathLike | None = None,
                data_root: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw, data_root)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
