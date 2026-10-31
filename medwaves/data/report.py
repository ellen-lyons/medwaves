"""Stage-1 data report: printed and written to DATA_ROOT/reports/stage1_<year>.md and .json."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from medwaves.data.common import Layout, write_json
from medwaves.data.qc import ALTIMETER_RULES, BUOY_RULES


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "-"


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)\n"
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples():
        lines.append("| " + " | ".join("-" if v is None or v != v else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _monthly_obs(ds: xr.Dataset, by_platform: bool = False) -> pd.DataFrame:
    if ds is None or ds.sizes.get("obs", 0) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({"month": pd.to_datetime(ds["step_time"].values).strftime("%Y-%m"),
                       "platform": ds["platform"].values})
    if by_platform:
        return pd.crosstab(df["month"], df["platform"])
    return df.groupby("month").size().to_frame("obs")


def build(year: int, layout: Layout, sections: dict) -> tuple[str, dict]:
    """sections: catalogue, mesh (Dataset), qc_buoys (dict), qc_altimeter (dict), superobs (dict),
    aligned_buoys, aligned_altimeter (Datasets), wind (Dataset), altimeter_manifests (dict month ->
    in-domain points per satellite), inventory (DataFrame), split (dict), stores (dict)."""
    md = [f"# Stage 1 data report: {year}\n"]
    js: dict = {"year": year}

    # Sources
    md.append("## Years available per source\n")
    rows = []
    for name, c in sections["catalogue"].items():
        now = c.get("catalogue_now")
        rows.append({"source": name, "start": c["start"], "end": c["end"], "role": c["role"],
                     "live catalogue": "-" if now is None else (f"{now[0]} -> {now[1]}"
                                                               if isinstance(now, (list, tuple)) else now)})
    md.append(_md_table(pd.DataFrame(rows).set_index("source")))
    missions = sections["catalogue"].get("altimeter_l3_my", {}).get("missions", {})
    if missions:
        md.append("\nAltimeter missions in the multi-year product: " + ", ".join(
            f"{m} {v['start'][:4]}-{v['end'][:4]}" for m, v in missions.items()) + "\n")
    js["catalogue"] = sections["catalogue"]

    # Mesh
    mesh = sections["mesh"]
    fetch = mesh["fetch_km"].values
    js["mesh"] = {"nodes": int(mesh.sizes["node"]), "directed_edges": int(mesh.sizes["edge"]),
                  "nodes_on_land": int(mesh["on_land"].sum()),
                  "edge_km_median": float(np.median(mesh["edge_km"])),
                  "depth_m_quantiles": np.quantile(mesh["depth_m"], [0, .5, 1]).round(1).tolist(),
                  "dist_coast_km_quantiles": np.quantile(mesh["dist_coast_km"], [0, .5, 1]).round(1).tolist(),
                  "fetch_km_quantiles": np.quantile(fetch, [0, .5, 1]).round(1).tolist(),
                  "fetch_share_at_cap": float(np.mean(fetch >= fetch.max()))}
    md.append("\n## Mesh\n")
    md.extend(f"- {k}: {v}\n" for k, v in js["mesh"].items())

    # Monthly counts
    qb, qa = sections["qc_buoys"], sections["qc_altimeter"]
    bm = pd.DataFrame(qb["by_month"]).T if qb.get("by_month") else pd.DataFrame()
    am = pd.DataFrame(qa["by_month"]).T if qa.get("by_month") else pd.DataFrame()
    monthly = pd.DataFrame(index=sorted(set(bm.index) | set(am.index)))
    monthly.index.name = "month"
    if not bm.empty:
        monthly["buoy raw (hourly)"] = bm["raw"]
        monthly["buoy after QC"] = bm["kept"]
    ab = _monthly_obs(sections["aligned_buoys"])
    if not ab.empty:
        monthly["buoy 3-h obs"] = ab["obs"]
    if not am.empty:
        monthly["alt raw (1 Hz)"] = am["raw"]
        monthly["alt after QC"] = am["kept"]
    aa = _monthly_obs(sections["aligned_altimeter"], by_platform=True)
    for m in aa.columns:
        monthly[f"super-obs {m}"] = aa[m]
    wind = sections["wind"]
    wm = pd.Series(pd.to_datetime(wind["time"].values).strftime("%Y-%m")).value_counts()
    monthly["ERA5 steps"] = wm
    monthly = monthly.fillna(0).astype(int)
    md.append("\n## Counts per source per month\n")
    md.append(_md_table(monthly))
    js["monthly"] = monthly.to_dict(orient="index")
    manifests = sections.get("altimeter_manifests", {})
    if manifests:
        sat = pd.DataFrame(manifests).T.fillna(0).astype(int).sort_index()
        sat.index.name = "month"
        md.append("\nAltimeter points inside the domain by mission, before keeping only the "
                  "configured missions:\n\n")
        md.append(_md_table(sat))
        js["altimeter_domain_points_by_mission"] = sat.to_dict(orient="index")
    md.append(f"\nWind: {wind.sizes['time']} steps, {wind.attrs.get('missing_steps', 0)} missing, "
              f"{int(np.isnan(wind['u10'].values).sum())} NaN values.\n")

    # QC
    md.append("\n## Quality control\n")
    for label, q, rules in (("Buoys (hourly points)", qb, BUOY_RULES),
                            ("Altimeter (1 Hz points)", qa, ALTIMETER_RULES)):
        n = q.get("n_raw", 0)
        removed = q.get("removed", {})
        total = sum(removed.values())
        md.append(f"\n**{label}**: {n} raw, {q.get('n_kept', 0)} kept, {_pct(total, n)} dropped.\n\n")
        t = pd.DataFrame({"points removed": [removed.get(r, 0) for r in rules],
                          "% of raw": [_pct(removed.get(r, 0), n) for r in rules]}, index=list(rules))
        t.index.name = "rule"
        md.append(_md_table(t))
    so = sections["superobs"]
    md.append(f"\nAltimeter super-obs: {so.get('superobs', 0)} kept; "
              f"{so.get('superobs_dropped_few_points', 0)} dropped for having fewer than the minimum "
              f"points ({so.get('points_in_dropped_superobs', 0)} points).\n")
    if not bm.empty:
        md.append("\nBuoy points removed per rule per month:\n\n")
        md.append(_md_table(bm[[r for r in BUOY_RULES if r in bm.columns]].astype(int)))
    bp = pd.DataFrame(qb.get("by_platform", {})).T
    if not bp.empty and "stuck" in bp:
        bp = bp.astype(int)
        worst = bp[bp["stuck"] > 0].assign(stuck_share=lambda d: (100 * d["stuck"] / d["raw"]).round(1))
        worst = worst.sort_values("stuck", ascending=False)[["raw", "stuck", "stuck_share", "kept"]].head(15)
        worst.index.name = "buoy"
        md.append(f"\nStuck-value rule: {int((bp['stuck'] > 0).sum())} of {len(bp)} buoys affected. "
                  "Most affected (stuck_share in %):\n\n")
        md.append(_md_table(worst))
    js["qc"] = {"buoys": {k: v for k, v in qb.items() if k != "by_platform"},
                "altimeter": qa, "superobs": so}

    # Split and stores
    inv, split, stores = sections["inventory"], sections["split"], sections["stores"]
    md.append("\n## Buoy split and stores\n")
    md.append(f"- Inventory (all configured years, after QC): {len(inv)} buoys in "
              f"{inv['cluster'].nunique() if len(inv) else 0} clusters\n")
    md.append(f"- Buoy split: {split['state']}\n")
    if split.get("train_pool") is not None:
        md.append(f"- Buoys per store (whole inventory): train_pool {split['train_pool']}, "
                  f"heldout {split['heldout']} ({_pct(split['heldout'], split['train_pool'] + split['heldout'])})\n")
        if len(inv):
            held = inv[inv["platform_id"].isin(split.get("heldout_ids", []))]
            md.append("- Held-out buoys per region: " + ", ".join(
                f"{r} {int((held['region'] == r).sum())}/{int((inv['region'] == r).sum())}"
                for r in sorted(inv["region"].unique())) + "\n")
    for role, c in stores.items():
        md.append(f"- Store {role}: {c['buoy']['obs']} buoy obs from {len(c['buoy']['platforms'])} buoys, "
                  f"{c['altimeter']['obs']} altimeter super-obs from {c['altimeter']['platforms'] or 'no mission'}\n")
    js["split"] = {k: v for k, v in split.items() if k != "heldout_ids"}
    js["stores"] = stores
    text = "".join(md)
    layout.reports.mkdir(parents=True, exist_ok=True)
    layout.report(year, "md").write_text(text)
    write_json(layout.report(year, "json"), js)
    return text, js

