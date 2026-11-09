# ARISE: Anchor-Reinforced Intrusion Detection System Enhancement Framework

Minimal code to reproduce **Table II** of the paper: three unsupervised target detectors
(HorusEye, KitNET, a plain autoencoder "AE") on four datasets (IoT, CIC-IDS2017, CIC-IoT2023, Kitsune),
before and after ARISE enhancement.

Names in the code and in the paper: the target `kitnet` is **Kitsune** in the paper and `simpleae` is **AE**; the
dataset `iot` is the **HorusEye** dataset, `cic2017` is CIC-IDS2017, `cic23` is CIC-IoT2023, and `kitsune` is the
Kitsune dataset.

## Layout

```
arise/                 ARISE
  cluster.py           knowledge mining (1): label aggregation + DBSCAN behaviour clustering
  anchor.py            knowledge mining (2): anchor selection, Eq. (1)
  train.py             knowledge integration: anchor-constrained fine-tuning of the target model
  test.py              detection test (PR-AUC, TPR at FPR <= 1%)
  base.py              trains a target model from scratch (not needed for Table II; base models are released)
  data.py, data_v1.py  data splits and test loaders;  targets.py  the three target models
targets/<target>/      target model code and released base models + scalers (params/<column>/)
configs/<column>.yaml  every setting of a Table II column (seed, splits, budgets, learning rates, ...)
checkpoints/           the released ARISE-enhanced model of every column (chosen.json: its alpha / R)
scripts/               run_column.sh, evaluate_released.sh, summarize.py, grids/
expected/table2.csv    the numbers reported in the paper
data/                  datasets (not included; see below)
```

A column is `<target>_<dataset>`, e.g. `horuseye_iot`, `kitnet_cic23`, `simpleae_kitsune`.

## Environment

Python 3.8 and the pinned versions in `requirements.txt`; one CUDA GPU. All runs use `SEED: 20`.

## Data

Place the feature files under `data/` so that the `normal_dir` / `attack_dir` entries of `configs/*.yaml` resolve
(they are relative to `targets/<target>/`):

| directory | used by | content |
|---|---|---|
| `data/horuseye/normal-kitsune_test/`, `data/horuseye/Anomaly/attack_kitsune/` | HorusEye, KitNET × IoT | IoT testbed benign / attack traffic |
| `data/horuseye/CIC-2017/attack/` | HorusEye, KitNET × CIC-IDS2017 | CIC-IDS2017 attacks (benign background: IoT testbed) |
| `data/horuseye/CIC-IoT-2023/{normal,anomaly}/` | HorusEye, KitNET × CIC-IoT2023 | CIC-IoT2023 |
| `data/horuseye/Kitsune/{normal,attack}/` | HorusEye × Kitsune | Kitsune dataset |
| `data/kitnet_kitsune/{normal,attack}/` | KitNET × Kitsune | Kitsune dataset (KitNET feature files) |
| `data/simpleae/{Horus,CIC-2017,CIC-IoT-2023,Kitsune}/{normal,attack}/` | AE × all | CICFlowMeter flow features |

Download links for the feature files will be added here.

## Quick check: evaluate the released models

```bash
scripts/evaluate_released.sh 0                 # GPU 0, all 12 columns
scripts/evaluate_released.sh 0 horuseye_iot    # one column
```

This scores the released base model and ARISE checkpoint of each column on its test split and prints them next to
the paper's numbers (`results/table2_released.csv`, column `max_abs_diff_vs_paper`).

## Full pipeline for one column

```bash
scripts/run_column.sh horuseye_iot 0
```

Steps: behaviour clustering → anchor selection (0.1% of each attack class, drawn from the validation split only) →
fine-tuning on the grid in `scripts/grids/<column>.txt` (`none:0:0` is benign-only fine-tuning;
`anchor:<alpha>:<R>` is ARISE) → summary (`results/table2_runs.csv`). For each configuration the epoch, and for the
column the (alpha, R), are chosen by PR-AUC on the selection set (benign validation + validation attacks), never on
the test set. On one RTX 3090, mining plus one 20-epoch fine-tuning run of HorusEye × IoT takes about 2 h;
the grid adds one fine-tuning run per line of the grid file.

Clustering is deterministic. Anchor selection and fine-tuning run on the GPU, so a re-run is not bit-identical:
re-running HorusEye × IoT from scratch reproduced the clusters exactly, 877 of the 878 anchors, and the Overall
PR-AUC / TPR of Table II to within 0.002 (per class within 0.003 PR-AUC and 0.02 TPR). The released checkpoints
reproduce Table II exactly.

Outputs per run: `results/<column>/runs/<run>/{test.csv, history.csv, run_info.json, model.pkl}`;
`run_info.json` records every resolved setting and the md5 of every input file.
Any setting can be overridden from the environment, e.g. `ARISE_FT_EPOCHS=10 scripts/run_column.sh ...`.
