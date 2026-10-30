# medwaves
Observation-only GNN for Mediterranean significant wave height. See `CLAUDE.md` for the brief.

## Local setup (CPU only, tiny samples)

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-local.txt
    .venv/bin/pytest
    .venv/bin/medwaves-check-splits --data-root ./scratch-data

## Colab

Open `colab/run.ipynb`, set the Colab Secrets listed in its first cell, and run top to bottom.
