# Data

This directory contains information and instructions for obtaining the datasets used in the experiments.

The repository evaluates the curvature-guided community detection framework on one synthetic benchmark and four real-world networks:

- Facebook
- Amazon
- DBLP
- YouTube

Raw real-world datasets are not duplicated in this repository. Instead, the corresponding experiment notebooks document the preprocessing and graph construction procedures used for each benchmark.

## Dataset Organization

After obtaining the required datasets, place the corresponding files in this directory or update the data paths in the experiment notebooks as needed.

The real-world experiments are implemented in:

- `notebooks/Facebook.ipynb`
- `notebooks/Amazon.ipynb`
- `notebooks/DBLP.ipynb`
- `notebooks/Youtube.ipynb`

The synthetic benchmark is generated programmatically and does not require an external data file:

- `notebooks/Synthetic_Module_Verification.ipynb`

## Reproducibility

Each notebook contains the dataset-specific preprocessing steps used to construct the graph and prepare the available ground-truth community labels for evaluation.

Additional dataset source information and download instructions will be included with the final reproducibility release.