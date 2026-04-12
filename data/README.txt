DATA DIRECTORY
==============

OVERVIEW
--------
This directory stores:
1) the cluster-labelled meta-training table, and
2) the ground-truth CSVs used for numbered adaptation cases.

data/train/
-----------
Put the meta-training CSV here.

The training loader accepts either the legacy names or the compact names.

Required logical fields:
- one of: E or E_raw
- one of: f or f_raw
- one of: k or k_raw
- Initial_L1
- Final_L2
- Avg_L3
- Cluster

Notes:
- Cluster labels must be numeric.
- Additional columns are allowed and ignored by the core pipeline.
- The task ordering logic uses the affinity-related columns and the cluster labels
  from this CSV.

Example accepted headers:
- E, f, k, Initial_L1, Final_L2, Avg_L3, Cluster
or
- E_raw, f_raw, k_raw, Initial_L1, Final_L2, Avg_L3, Cluster

data/eval/
----------
Put the ground-truth CSVs for numbered adaptation cases here.

Preferred naming:
- sample_task1.csv
- sample_task2.csv
- ...
- sample_task10.csv

Supported coordinate aliases:
- x, y
- X, Y
- coord_x, coord_y

Supported displacement aliases:
- u, v
- ux, uy
- disp_x, disp_y

The adaptation code resolves the correct CSV automatically from task.case_index.

The evaluation CSV may be either:
- a standard row-wise numeric table, or
- a single-row file containing Python-style lists.
