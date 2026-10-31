"""Command-line entry points, one per stage. Stages 2-4 are not implemented yet."""
from __future__ import annotations

import argparse
import sys

from medwaves.config import DEFAULT_CONFIG, Config, load_config


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--data-root", default=None,
                        help="override paths.data_root (also settable via $MEDWAVES_DATA_ROOT)")
    return parser


def _load(description: str, argv: list[str] | None) -> Config:
    args = _parser(description).parse_args(argv)
    return load_config(args.config, args.data_root)


def _stub(stage: str, description: str, argv: list[str] | None) -> int:
    cfg = _load(description, argv)
    print(f"[medwaves-{stage}] data_root: {cfg.paths.data_root}")
    print(f"[medwaves-{stage}] not implemented yet", file=sys.stderr)
    return 2


def data(argv: list[str] | None = None) -> int:
    from medwaves.data.pipeline import STEPS, Stage1, probe

    parser = _parser("Stage 1: download, quality-filter and align observations and winds. "
                     "Re-runnable: every step skips work already done.")
    parser.add_argument("--only", choices=STEPS, action="append",
                        help="run only this step (repeatable); default: all, in order")
    parser.add_argument("--sample-days", type=int, default=None,
                        help="use only the first N days of each year (laptop tests)")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate fake raw data instead of downloading (no network); "
                             "implies --sample-days pilot.local_sample_days unless given")
    parser.add_argument("--probe", action="store_true",
                        help="download one day of each source into raw/probe/ and print its format")
    parser.add_argument("--offline", action="store_true", help="skip the live catalogue lookup")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_root)
    print(f"[medwaves-data] data_root: {cfg.paths.data_root}")
    if args.probe:
        probe(cfg)
        return 0
    sample_days = args.sample_days
    if args.synthetic:
        from medwaves.data.synthetic import write_raw

        sample_days = sample_days or cfg.pilot.local_sample_days
        write_raw(cfg, sample_days)
    Stage1(cfg, sample_days, live_catalogue=not (args.offline or args.synthetic)).run(args.only or STEPS)
    return 0


def baselines(argv: list[str] | None = None) -> int:
    return _stub("baselines", "Stage 2: baselines and reanalysis comparison.", argv)


def train(argv: list[str] | None = None) -> int:
    return _stub("train", "Stage 3: train the GNN (wind-only, then with observations).", argv)


def evaluate(argv: list[str] | None = None) -> int:
    return _stub("eval", "Stage 4: shared evaluation of all models and baselines.", argv)


def check_splits(argv: list[str] | None = None) -> int:
    from medwaves.splits import SpaceSplit, TimeSplit, draft_buoy_split_path

    cfg = _load("Print the resolved time and space splits.", argv)
    time_split = TimeSplit.from_config(cfg)
    space = SpaceSplit.from_config(cfg)
    wind_h = time_split.wind_history.total_seconds() / 3600
    obs_h = time_split.obs_history.total_seconds() / 3600
    print(f"data_root: {cfg.paths.data_root}")
    print(f"reading processed data from: {cfg.paths.processed_read}")
    print(f"time windows, half-open UTC (lookback: wind {wind_h:g} h, observations {obs_h:g} h):")
    for name, w in time_split.windows.items():
        print(f"  {name:<5} {w.start:%Y-%m-%d} -> {w.end:%Y-%m-%d}")
    if space.buoys_frozen:
        print(f"buoys: {len(space.train_buoys)} train pool, {len(space.heldout_buoys)} held out")
    else:
        draft = draft_buoy_split_path(cfg)
        state = (f"a draft is waiting at {draft}: copy it to {cfg.splits.space.heldout_buoys_file} "
                 "and commit" if draft.exists() else "not drawn yet (stage 1)")
        print(f"buoys: {cfg.splits.space.heldout_buoys_file} not in the repo; {state}. "
              "Every buoy is refused until it is committed.")
    print(f"held-out mission: {space.heldout_mission or 'not set'}; "
          f"train missions: {sorted(space.train_missions) or 'none'}; pilot: {cfg.pilot.mission}")
    print(f"masking fraction: {cfg.splits.masking_fraction}")
    unverified = [n for n, p in cfg.products.items() if not p.verified]
    print(f"unverified products: {', '.join(unverified) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(check_splits())
