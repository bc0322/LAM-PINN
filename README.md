# LAM-PINN

> [!CAUTION]
> A reproducibility issue was identified and corrected.

## 🚨 **Notice (Apr 13, 2026)**
**Please use the updated code and instructions in this repository.**

This repository is a cleaned public release of the **core LAM-PINN pipeline** used in the manuscript.  
It is intentionally organized as a compact, reproducible reference implementation rather than a dump of research notebooks.

## What is included

- task-table ingestion from CSV
- affinity-aware task ordering and cluster-wise compositional meta-training
- checkpointing and run logging
- single-task adaptation from a trained checkpoint
- quantitative evaluation against a ground-truth CSV
- deformation-field visualization in a publication-friendly format

The current public package exposes the main training/adaptation workflow in a way that can be extended to additional benchmarks without changing the repository layout.

---

## Repository layout

```text
lam-pinn-public/
├── configs/
│   ├── train.yaml
│   └── adapt.yaml
├── data/
│   ├── README.txt
│   ├── train/
│   └── eval/
├── lam_pinn/
│   ├── cli/
│   ├── data/
│   ├── engine/
│   ├── evaluation/
│   ├── models/
│   ├── physics/
│   ├── pretrained/
│   └── utils/
├── scripts/
│   ├── run_train.py
│   └── run_adapt.py
├── outputs/
├── README.md
├── README.txt
├── requirements.txt
└── pyproject.toml
```

---

## Installation

Python 3.8+ is supported.
Run from the repository root:

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

---

## Input CSV format

### 1) Meta-training task table

Place the training CSV in `data/train/` and point `configs/train.yaml` to it.

Required columns:

- one of: `E` or `E_raw`
- one of: `f` or `f_raw`
- one of: `k` or `k_raw`
- `Initial_L1`
- `Final_L2`
- `Avg_L3`
- `Cluster`

The loader accepts either the legacy names (`E_raw`, `f_raw`, `k_raw`) or the compact names (`E`, `f`, `k`) and canonicalizes them internally.
Additional columns are allowed and ignored by the core pipeline.

### 2) Evaluation CSVs for numbered adaptation cases

Place the evaluation CSVs in `data/eval/` using the preferred naming pattern:

- `sample_task1.csv`
- `sample_task2.csv`
- ...
- `sample_task10.csv`

The adaptation pipeline can automatically resolve the correct CSV and task parameters from a single case index.

For backward compatibility with the current archive layout, case 1 may also be loaded from `sample_task.csv` if `sample_task1.csv` is not present.

Expected displacement columns are recognized case-insensitively from aliases such as:

- coordinates: `x`, `y`, `X`, `Y`, `coord_x`, `coord_y`
- displacement: `u`, `v`, `ux`, `uy`, `disp_x`, `disp_y`

Both of the following formats are supported:

- row-wise numeric table
- a single-row CSV whose cells contain Python-style lists

---

## Quick start

### Step 1. Configure meta-training

Edit:

- `configs/train.yaml`

Then run:

```bash
PYTHONPATH=. python scripts/run_train.py --config configs/train.yaml
```

### Step 2. Configure single-task adaptation by case index

Set `task.case_index` in `configs/adapt.yaml`, or override it from the command line.

Examples:

```bash
PYTHONPATH=. python scripts/run_adapt.py --config configs/adapt.yaml
PYTHONPATH=. python scripts/run_adapt.py --config configs/adapt.yaml --case-index 3
```

---

## Main outputs

A training run creates a timestamp-safe run directory under `outputs/train/` containing:

- `checkpoints/model.pt`
- `artifacts/balanced_training_order.csv`
- `artifacts/training_trace.csv`
- `artifacts/cluster_mapping.json`
- `logs/run.log`
- `training_summary.json`

An adaptation run creates a run directory under `outputs/adapt/` containing:

- `checkpoints/adapted_model.pt`
- `artifacts/adaptation_trace.csv`
- `artifacts/prediction_comparison.csv`
- `figures/deformation_comparison.png`
- `figures/loss_curve.png`
- `figures/gate_trajectory.png`
- `logs/run.log`
- `adaptation_summary.json`

---

## Notes

- The public release focuses on the **clean LAM-PINN reference path**: CSV ingestion → cluster-aware meta-training → checkpointing → single-task adaptation → evaluation/visualization.
- Research-only notebook utilities and intermediate experiments were intentionally left out to keep the release concise and reproducible.
- To extend this release to another PDE family, the main entry points are `lam_pinn/physics/` and the task-property construction in `lam_pinn/data/ingestion.py`.
