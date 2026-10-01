# Curvature-Guided Community Detection

<!-- TODO: Replace this paragraph with a concise summary of the project, its contribution, and the problem it addresses. -->

This repository contains research code and experiment artifacts for curvature-guided community detection.

## Project Status

<!-- TODO: State whether this repository supports reproducing a paper, ongoing research, or general use. -->

This is research software. Results and workflows may depend on the datasets, package versions, random seeds, and settings documented below.

## Overview

<!-- TODO: Briefly explain the method, baselines, and evaluation metrics. Link to the paper or preprint when available. -->

## Repository Structure

```text
.
|-- Amazon.ipynb
|-- DBLP.ipynb
|-- Facebook.ipynb
|-- Youtube.ipynb
|-- Synthetic_Module_Verification.ipynb
|-- community_detection_utils.py
|-- environment.yml
|-- Template_for_Journal_of_Complex_Networks__COMNET___1_-3.pdf
|-- data/                         # Created locally as datasets are downloaded
|   |-- amazon/
|   |-- dblp/
|   |-- facebook/
|   `-- youtube/
|-- figures/
|   |-- amazon/
|   |-- dblp/
|   |-- facebook/
|   |-- youtube/
|   `-- synthetic_module_verification/  # Created when its notebook runs
`-- results/
  |-- amazon/                   # Existing result files; reruns create:
  |   |-- tables/
  |   `-- sensitivity/
  |-- dblp/
  |   |-- tables/
  |   `-- sensitivity/
  |-- facebook/
  |   |-- tables/
  |   `-- sensitivity/
  |-- youtube/
  |   |-- tables/
  |   `-- sensitivity/
  `-- synthetic_module_verification/  # Created when its notebook runs
    |-- tables/
    `-- sensitivity/
```

## Requirements

- Python: <!-- TODO: Add the tested Python version or supported version range. -->
- Dependencies: <!-- TODO: Link to `requirements.txt`, `environment.yml`, or another maintained environment file. -->
- Resources: <!-- TODO: Note expected memory, runtime, and hardware, especially for the larger datasets. -->

## Installation

<!-- TODO: Add tested setup commands after choosing and adding a dependency manifest. -->

```bash
# TODO: Create and activate the project's environment.
# TODO: Install dependencies from the dependency manifest.
```

## Data

Raw datasets are not included in this repository. The experiment notebooks download or expect dataset files under local `data/` directories.

<!-- TODO: For each dataset, provide its source URL, exact file/version, preprocessing steps, license or terms, and citation. State whether notebooks download it automatically or require manual download. -->

| Dataset | Source and version | Local path | Citation / terms |
| --- | --- | --- | --- |
| Amazon | TODO | `data/amazon/` | TODO |
| DBLP | TODO | `data/dblp/` | TODO |
| Facebook | TODO | `data/facebook/` | TODO |
| YouTube | TODO | `data/youtube/` | TODO |

## Reproducing Experiments

Each notebook discovers the repository root from `environment.yml` and writes generated files to the shared directories listed under [Results](#results). Run cells from top to bottom; the data-loading cells download datasets into `data/<dataset>/` when they are not already present.

1. Complete the installation steps above.
2. Obtain the datasets as described in [Data](#data).
3. Open the notebook for the dataset or experiment of interest and run its cells in order.
4. Find generated CSV tables, sensitivity results, and figures in the shared output directories below.

Random seeds, preprocessing choices, algorithm parameters, and evaluation details: <!-- TODO: document or link to the exact configuration used for reported results. -->

## Results

The notebooks create these output directories as needed. Existing committed result files may remain directly inside their dataset's `results/` folder; new runs use:

- `results/<dataset>/tables/` contains result, structural, and complexity CSV/LaTeX tables.
- `results/<dataset>/sensitivity/` contains raw and summarized sensitivity CSV files.
- `figures/<dataset>/` contains generated PNG and PDF figures, including heatmaps.

For the synthetic verification notebook, `<dataset>` is `synthetic_module_verification`.

<!-- TODO: Identify the main result files and explain which experiment/configuration produced them. Add links to the paper and selected figures if useful. -->

## Citation

<!-- TODO: Replace with the paper citation. Add a CITATION.cff file for GitHub's citation support when publication details are available. -->

```bibtex
@misc{TODO,
  title        = {TODO},
  author       = {TODO},
  year         = {TODO},
  howpublished = {TODO},
  doi          = {TODO}
}
```

Please also cite the original sources of any datasets used.

## License

<!-- TODO: Choose a license for this repository and add its license file. Until then, reuse permissions are not specified. -->

## Contact

<!-- TODO: Add a maintainer contact, lab page, or issue-reporting link. -->
