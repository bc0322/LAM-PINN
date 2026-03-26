DATA DIRECTORY
==============

data/train/
-----------
Put the cluster-labelled meta-training table here.
Required columns:
- E_raw
- f_raw
- k_raw
- Initial_L1
- Final_L2
- Avg_L3
- Cluster

data/eval/
----------
Put the ground-truth CSVs for numbered adaptation cases here.

Preferred naming:
- sample_task1.csv
- sample_task2.csv
- ...
- sample_task10.csv

Supported column aliases:
- x/y: x, y, X, Y, coord_x, coord_y
- u/v: u, v, ux, uy, disp_x, disp_y

The adaptation code resolves the correct CSV automatically from task.case_index.
For backward compatibility with the current archive layout, case 1 may also use sample_task.csv.

The evaluation CSV may be either:
- a standard row-wise table, or
- a single-row file containing Python-style lists.
