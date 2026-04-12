LAM-PINN PUBLIC RELEASE
=======================

NOTICE (Apr 12, 2026)
---------------------
A reproducibility issue was identified and corrected.
Please use the updated code and instructions in this repository.

This repository is a cleaned public release of the core LAM-PINN pipeline used in the manuscript.
It is organized as a compact, reproducible Python package rather than a notebook collection.

INCLUDED
--------
1) task-table ingestion from CSV
2) affinity-aware task ordering and cluster-wise compositional meta-training
3) checkpointing and run logging
4) single-task adaptation from a trained checkpoint
5) quantitative evaluation against a ground-truth CSV
6) deformation-field visualization in a publication-friendly format

QUICK START
-----------
1) Put the training CSV in data/train/
2) Put the evaluation CSV in data/eval/
3) Edit configs/train.yaml and set task.case_index in configs/adapt.yaml
4) Run from the repository root:

   python scripts/run_train.py --config configs/train.yaml
   python scripts/run_adapt.py --config configs/adapt.yaml

Optional case override from the command line:

   python scripts/run_adapt.py --config configs/adapt.yaml --case-index 3

REQUIRED TRAINING CSV COLUMNS
-----------------------------
E_raw, f_raw, k_raw, Initial_L1, Final_L2, Avg_L3, Cluster

REQUIRED EVALUATION CSV CONTENT
-------------------------------
Preferred file naming in data/eval/:
- sample_task1.csv
- sample_task2.csv
- ...
- sample_task10.csv

Coordinate columns: x/y (or X/Y, coord_x/coord_y)
Displacement columns: u/v (or ux/uy, disp_x, disp_y)

Case index automatically resolves the matching CSV and the corresponding (E, f, k) parameters.
For backward compatibility, case 1 may also use data/eval/sample_task.csv.

OUTPUTS
-------
Training:
- checkpoints/model.pt
- artifacts/balanced_training_order.csv
- artifacts/training_trace.csv
- artifacts/cluster_mapping.json
- logs/run.log
- training_summary.json

Adaptation:
- checkpoints/adapted_model.pt
- artifacts/adaptation_trace.csv
- artifacts/prediction_comparison.csv
- figures/deformation_comparison.png
- figures/loss_curve.png
- figures/gate_trajectory.png
- logs/run.log
- adaptation_summary.json

NOTE
----
This public release intentionally focuses on the clean LAM-PINN reference path:
CSV ingestion -> cluster-aware meta-training -> checkpointing -> single-task adaptation -> evaluation/visualization.
Research-only notebook utilities and intermediate experimental branches were left out on purpose.
