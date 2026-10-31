# medwaves
Observation-only GNN for Mediterranean significant wave height. See `CLAUDE.md` for the brief.

## Local setup (CPU only, tiny samples)

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-local.txt
    .venv/bin/pytest
    .venv/bin/medwaves-check-splits --data-root ./scratch-data
    .venv/bin/medwaves-data --data-root ./scratch-data --synthetic   # stage 1 on fake data, no network

## Stage 1 (data pipeline)

`medwaves-data` runs download, mesh, inventory, process, stores and report in order; each step skips
work already done. `--probe` downloads one day of each source and prints its format; run it once on
Colab before the full run. The held-out buoy split is drawn once to `DATA_ROOT/splits/` and must be
committed as `configs/heldout_buoys.yaml` before buoys enter the stores (see the notebook).

## Colab

Open `colab/run.ipynb`, set the Colab Secrets listed in its first cell, and run top to bottom.
