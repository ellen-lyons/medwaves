"""Time and space splits, and the guards that keep held-out observations out of model inputs.

Rules enforced here:
- Train/val/test windows are disjoint, with a gap of at least `gap_days` where they touch. A sample
  is only valid if both its wind lookback and its observation lookback lie inside one window, and
  every observation fed as input must itself fall inside that window.
- Every observation platform (buoy ID or altimeter mission) has a role, TRAIN_POOL or HELDOUT.
  Unknown platforms raise: the split fails closed, never open.
- HELDOUT observations may be read only as evaluation targets, never as model inputs.
- The buoy split is drawn once (stage 1), written to DATA_ROOT/splits/ as a draft, and becomes
  active only when the user copies it into the repo and commits it. It is never redrawn silently.
"""
from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from medwaves.config import REPO_ROOT, Config, Paths

SPLIT_NAMES = ("train", "val", "test")
BUOY = "buoy"
ALTIMETER = "altimeter"
INPUT = "input"
TARGET = "target"


class LeakageError(RuntimeError):
    """A held-out or unknown observation is about to reach a model input."""


class ObsRole(str, enum.Enum):
    TRAIN_POOL = "train_pool"
    HELDOUT = "heldout"


@dataclass(frozen=True)
class Window:
    """Half-open time window [start, end), naive UTC."""
    start: dt.datetime
    end: dt.datetime

    def contains(self, t: dt.datetime) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class TimeSplit:
    windows: dict[str, Window]
    wind_history: dt.timedelta
    obs_history: dt.timedelta

    @classmethod
    def from_config(cls, cfg: Config) -> TimeSplit:
        gap = dt.timedelta(days=cfg.splits.gap_days)
        windows: dict[str, Window] = {}
        prev_end = None
        for name in SPLIT_NAMES:
            interval = getattr(cfg.splits, name)
            start = dt.datetime.combine(interval.start, dt.time())
            end = dt.datetime.combine(interval.end, dt.time())
            # Where splits touch, the gap is cut from the start of the later split.
            if prev_end is not None and start - prev_end < gap:
                start = prev_end + gap
            if start >= end:
                raise ValueError(f"split {name} is empty after removing the gap")
            windows[name] = Window(start, end)
            prev_end = end
        return cls(windows,
                   dt.timedelta(hours=cfg.time.wind_history_hours),
                   dt.timedelta(hours=cfg.time.obs_history_hours))

    def which(self, t: dt.datetime) -> str:
        """Name of the split containing time t, or "gap" if none does."""
        for name, window in self.windows.items():
            if window.contains(t):
                return name
        return "gap"

    def sample_ok(self, t: dt.datetime, split: str) -> bool:
        """True if a sample at t, with its full wind AND observation lookbacks, lies inside `split`."""
        window = self.windows[split]
        return (window.contains(t)
                and window.contains(t - self.wind_history)
                and window.contains(t - self.obs_history))

    def obs_input_ok(self, t_obs: dt.datetime, t: dt.datetime, split: str) -> bool:
        """True if an observation at t_obs may be an input to the sample at t in `split`."""
        return (self.sample_ok(t, split)
                and t - self.obs_history <= t_obs <= t
                and self.windows[split].contains(t_obs))


@dataclass(frozen=True)
class SpaceSplit:
    train_buoys: frozenset[str]
    heldout_buoys: frozenset[str]
    train_missions: frozenset[str]
    heldout_mission: str | None
    buoys_frozen: bool = True

    def __post_init__(self):
        if self.train_buoys & self.heldout_buoys:
            raise LeakageError(f"buoys in both train pool and held-out set: "
                               f"{sorted(self.train_buoys & self.heldout_buoys)}")
        if self.heldout_mission is not None and self.heldout_mission in self.train_missions:
            raise LeakageError(f"held-out mission {self.heldout_mission!r} is also a train mission")

    @classmethod
    def from_config(cls, cfg: Config) -> SpaceSplit:
        space = cfg.splits.space
        path = REPO_ROOT / space.heldout_buoys_file
        if path.exists():
            with open(path) as f:
                frozen = yaml.safe_load(f)
            train = frozenset(map(str, frozen["train_pool"]))
            heldout = frozenset(map(str, frozen["heldout"]))
            buoys_frozen = True
        else:
            # Not committed yet: every buoy is unknown, so role() refuses them all.
            train, heldout, buoys_frozen = frozenset(), frozenset(), False
        return cls(train, heldout, frozenset(space.train_missions), space.heldout_mission,
                   buoys_frozen)

    def role(self, source: str, platform_id: str) -> ObsRole:
        if source == BUOY:
            platform_id = str(platform_id)
            if platform_id in self.heldout_buoys:
                return ObsRole.HELDOUT
            if platform_id in self.train_buoys:
                return ObsRole.TRAIN_POOL
            raise LeakageError(f"buoy {platform_id!r} is in neither the train pool nor the held-out "
                               "set; update the committed buoy split explicitly")
        if source == ALTIMETER:
            if platform_id == self.heldout_mission:
                return ObsRole.HELDOUT
            if platform_id in self.train_missions:
                return ObsRole.TRAIN_POOL
            raise LeakageError(f"altimeter mission {platform_id!r} has no role in the config")
        raise ValueError(f"unknown observation source {source!r}")

    def assert_no_heldout(self, buoy_ids: Iterable[str] = (), missions: Iterable[str] = ()) -> None:
        """Raise unless every input platform is in the train pool. Call on every batch's inputs."""
        for buoy in set(map(str, buoy_ids)):
            if self.role(BUOY, buoy) is not ObsRole.TRAIN_POOL:
                raise LeakageError(f"held-out buoy {buoy!r} in model inputs")
        for mission in set(missions):
            if self.role(ALTIMETER, mission) is not ObsRole.TRAIN_POOL:
                raise LeakageError(f"held-out mission {mission!r} in model inputs")


def draft_buoy_split_path(cfg: Config) -> Path:
    """Where stage 1 writes the drawn buoy split, for the user to copy into the repo."""
    return cfg.paths.draft_splits / Path(cfg.splits.space.heldout_buoys_file).name


def write_draft_buoy_split(cfg: Config, train_pool: Iterable[str], heldout: Iterable[str],
                           seed: int, inventory_hash: str) -> Path:
    """Write a drawn buoy split to DATA_ROOT/splits/. Refuses if a split already exists anywhere."""
    committed = REPO_ROOT / cfg.splits.space.heldout_buoys_file
    draft = draft_buoy_split_path(cfg)
    for existing in (committed, draft):
        if existing.exists():
            raise FileExistsError(f"{existing} already exists; the buoy split is never redrawn "
                                  "silently. Delete it by hand to redraw.")
    split = SpaceSplit(frozenset(map(str, train_pool)), frozenset(map(str, heldout)),
                       frozenset(), None)
    draft.parent.mkdir(parents=True, exist_ok=True)
    with open(draft, "w") as f:
        yaml.safe_dump({"seed": seed, "inventory_hash": inventory_hash,
                        "train_pool": sorted(split.train_buoys),
                        "heldout": sorted(split.heldout_buoys)}, f)
    return draft


def obs_store(paths: Paths, role: ObsRole, purpose: str) -> Path:
    """Directory to read the `role` observations from. Held-out ones can only be read as targets."""
    if purpose not in (INPUT, TARGET):
        raise ValueError(f"purpose must be {INPUT!r} or {TARGET!r}")
    role = ObsRole(role)
    if purpose == INPUT and role is not ObsRole.TRAIN_POOL:
        raise LeakageError("held-out observations must never be read as model inputs")
    return paths.processed_read / "obs" / role.value
