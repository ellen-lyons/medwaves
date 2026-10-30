# medwaves: Learning Mediterranean waves from observations alone

## Project brief

A small graph neural network (GNN) that maps Mediterranean significant wave height using only real
observations (buoys, satellite altimeters) plus ERA5 winds. No physics-model output is ever used as a
training target. The question: how close can an observation-only model get to the physics reanalysis,
and where does it fail?

Main question: can a GNN trained only on observations reconstruct wave height at buoys and satellite
passes it has never seen, in a year it has never seen?

Follow-up questions:
- How much do live observations add beyond wind history alone?
- Where does it beat or lose to the physics reanalysis (open sea, coasts, swell)?

Background: WaveGraph (CMCC / University of Bologna, arXiv 2608.16449) emulates a Mediterranean wave
model with a GNN but learns from model output. Training on observations alone is named as a next step.
ECMWF's GraphDOP showed the idea works for weather.

## Design

| Part | Choice |
|---|---|
| Data | Copernicus Marine Med in-situ buoys, Copernicus L3 altimeter wave height, ERA5 10 m winds, 2019–2024 |
| Time step | 3-hourly |
| Graph | WaveGraph's ~60 km Mediterranean mesh if released; otherwise build an equivalent sea-only mesh |
| Node inputs | Last 24 h of wind, bathymetry, distance to coast. NO absolute coordinates |
| Observation inputs | Recent buoy and satellite values, tagged with age and type, linked to the 3 nearest nodes |
| Model | Small Interaction Network: latent size 64, 6–8 message-passing steps |
| Output | Wave height, read out only where an observation exists (no full field needed for training) |

Version 1 predicts wave height only; period and direction come later.

## Evaluation (must be honest; leakage is the main thing to avoid)

- Time split: train 2019–2022, validate 2023, test 2024, with a one-week gap at each boundary.
- Space split: about 20–25% of buoys and one whole satellite mission are held out. They are never used
  in training, not even as inputs.
- Masking in training: at each step, 25% of observations are hidden from the input and used as targets.
- Satellite QC: drop altimeter points within 15 km of land and points with bad quality flags.

Baselines:
1. Climatology (mean for that month and location)
2. Distance-weighted interpolation of nearby observations
3. Same model without observation inputs (wind only)
4. Referee: Copernicus Mediterranean wave reanalysis, at test points and as a full map

Metrics: RMSE, bias, scatter index, correlation. Report overall and for the top 5% of wave heights,
separately for held-out buoys and held-out satellite passes. All models and baselines use one shared
evaluation module.

Success: minimum = beat all three baselines on held-out buoys in 2024. Good = within reach of the
reanalysis near buoys, with an explanation of where and why it falls short. The wind-only vs full-model
gap measures the value of live observations. Failure cases are findings too.

## Plan (four stages, each usable on its own)

1. Data pipeline: download, quality-filter and align buoys, altimetry and winds to the mesh and 3-hourly steps.
   Start with ONE year (2019) and ONE altimeter mission, then scale up.
2. Baselines and reanalysis comparison, so the bar is known before training.
3. Model: wind-only version first, then add observation inputs.
4. Evaluation and write-up: scores, error maps, short report.

## Environment

- Code is written and lightly tested on a Linux Mint laptop, then run on Google Colab with a T4 GPU.
- The laptop is low on disk space and has no GPU: locally, install CPU-only PyTorch and test only on a
  tiny sample (a few days of data). Full downloads happen on Colab only.
- Repo: github.com/ellen-lyons/medwaves. Package `medwaves/` with CLI entry points per stage, plus one
  driver notebook `colab/run.ipynb` that mounts Google Drive, clones/pulls this repo, installs deps,
  reads credentials from Colab Secrets (google.colab.userdata), and runs each stage.
- All data, caches and checkpoints live under a configurable DATA_ROOT
  (default /content/drive/MyDrive/medwaves on Colab).
- Colab sessions die: every stage must be re-runnable and skip work already done; training must
  checkpoint every epoch and resume automatically.
- One YAML config for years, splits, mesh, model sizes and paths. Fixed seeds.
- Python with PyTorch and PyTorch Geometric. Data access via the `copernicusmarine` toolbox and `cdsapi`.

## Rules

- Held-out buoys and the held-out satellite mission must never appear in any training input. Keep tests
  that check this.
- Never use reanalysis or any physics-model output as a training target or input.
- Don't guess Copernicus product IDs or dataset names: look them up, verify them, and record them in the config.
- Never hard-code credentials.
- Don't download large data on the local machine.
- Work one stage at a time; don't build later stages until the current one runs and its data report looks sane.