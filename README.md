# Curvature-Guided Graph Sparsification for Community Detection

This repository contains code and experiment notebooks for studying curvature-guided graph sparsification as a way to reduce the cost of community detection while preserving community structure.

## Project Status

This repository accompanies the paper in progress "Curvature-Guided Graph Sparsification for Community Detection," which is being prepared for publication.
The notebooks and utilities are intended to reproduce the paper's reported experiments.
Results may depend on the datasets, package versions, random seeds, and settings documented below.

## Overview

The research evaluates a two-stage sparsification pipeline. First, lower Ricci curvature (LRC), a local combinatorial estimate, filters low-curvature edges. Ollivier-Ricci curvature (ORC) and discrete Ricci flow are then applied to the reduced graph before community detection. The experiments compare LRC-only and ORC-only baselines with the combined LRC-to-ORC method. Louvain is the primary detector; the notebooks also include Leiden comparisons.

The real-world notebooks build labeled benchmark graphs from SNAP's Amazon co-purchase, DBLP collaboration, Facebook ego-network, and YouTube social-network datasets. Because several datasets have overlapping communities, the notebooks select benchmark subsets and assign one ground-truth label per node for ARI/NMI evaluation. They report clustering scores, runtime, graph reduction, and structural summaries, and include parameter-sensitivity analyses with heatmaps. The synthetic notebook generates stochastic block model (SBM) graphs, checks the shared utility implementations against the original experiment, and explores parameter sensitivity.

Notebook guide:

- [Amazon.ipynb](Amazon.ipynb), [DBLP.ipynb](DBLP.ipynb), [Facebook.ipynb](Facebook.ipynb), and [Youtube.ipynb](Youtube.ipynb) construct dataset-specific benchmarks, run the curvature and community-detection comparisons, and analyze structural properties and sensitivity.
- [Synthetic_Module_Verification.ipynb](Synthetic_Module_Verification.ipynb) verifies the modularized pipeline on controlled SBM graphs and includes additional sensitivity experiments.
- [community_detection_utils.py](community_detection_utils.py) provides shared dataset-loading, graph-processing, curvature, community-detection, and evaluation functions used by the notebooks.

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

- Python 3.11, as specified in [`environment.yml`](environment.yml).
- Conda, with packages resolved from `conda-forge` and pip dependencies installed by the environment file.
- Core dependencies include Jupyter/IPykernel, NumPy, pandas, SciPy, NetworkX, Networkit, igraph/Leiden, scikit-learn, Matplotlib, and tqdm. The environment file also installs `GraphRicciCurvature` and `python-louvain` through pip.
- No minimum hardware requirements are specified. Memory use and runtime depend on the selected dataset and analysis; the larger real-world graph experiments may be resource-intensive.

## Installation

Create and activate the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate cg_community_detection_env
jupyter lab
```

In VS Code, select `cg_community_detection_env` as the notebook kernel.

## Data

Raw datasets are not included in this repository. The experiment notebooks download or expect dataset files under local `data/` directories.

The notebooks download the listed files into the local paths below when they are missing. Dataset terms are governed by the sources linked here; the manuscript does not state separate dataset licenses.

| Dataset | Source files | Local path | BibTeX keys |
| --- | --- | --- | --- |
| Amazon | [`com-amazon.ungraph.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz); [`com-amazon.top5000.cmty.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz) | `data/amazon/` | [snapAmazon2024](#snapamazon2024); [YangLeskovec2012GroundTruth](#yangleskovec2012groundtruth) |
| DBLP | [`com-dblp.ungraph.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz); [`com-dblp.top5000.cmty.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz) | `data/dblp/` | [snapDBLP2024](#snapdblp2024); [YangLeskovec2012GroundTruth](#yangleskovec2012groundtruth) |
| Facebook | [`facebook_combined.txt.gz`](https://snap.stanford.edu/data/facebook_combined.txt.gz); [`facebook.tar.gz`](https://snap.stanford.edu/data/facebook.tar.gz) (circles) | `data/facebook/` | [snapFacebook2012](#snapfacebook2012); [LeskovecMcAuley2012SocialCircles](#leskovecmcauley2012socialcircles) |
| YouTube | [`com-youtube.ungraph.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz); [`com-youtube.top5000.cmty.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz) | `data/youtube/` | [snapYouTube2009](#snapyoutube2009); [LeskovecEtAl2009CommunityStructure](#leskovecetal2009communitystructure) |

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

### Dataset References

The keys linked in the [Data](#data) table jump directly to the corresponding BibTeX entry.

<a id="leskovecetal2009communitystructure"></a>
```bibtex
@article{LeskovecEtAl2009CommunityStructure,
  author  = {Leskovec, Jure and Lang, Kevin J. and Dasgupta, Anirban and Mahoney, Michael W.},
  title   = {{Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters}},
  journal = {Internet Mathematics},
  year    = {2009},
  volume  = {6},
  number  = {1},
  pages   = {29--123}
}
```

<a id="leskovecmcauley2012socialcircles"></a>
```bibtex
@inproceedings{LeskovecMcAuley2012SocialCircles,
  author    = {Leskovec, Jure and McAuley, Julian J.},
  title     = {Learning to Discover Social Circles in Ego Networks},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2012}
}
```

<a id="snapamazon2024"></a>
```bibtex
@misc{snapAmazon2024,
  author       = {{Stanford Network Analysis Project}},
  title        = {Amazon Product Co-purchasing Network Metadata},
  year         = {2024},
  howpublished = {SNAP},
  url          = {https://snap.stanford.edu/data/com-Amazon.html},
  note         = {Accessed 2026-07-03}
}
```

<a id="snapdblp2024"></a>
```bibtex
@misc{snapDBLP2024,
  author       = {{Stanford Network Analysis Project}},
  title        = {DBLP Collaboration Network Metadata},
  year         = {2024},
  howpublished = {SNAP},
  url          = {https://snap.stanford.edu/data/com-DBLP.html},
  note         = {Accessed 2026-07-03}
}
```

<a id="snapyoutube2009"></a>
```bibtex
@misc{snapYouTube2009,
  author       = {{Stanford Network Analysis Project}},
  title        = {YouTube Social Network Dataset},
  year         = {2009},
  howpublished = {SNAP},
  url          = {https://snap.stanford.edu/data/com-Youtube.html},
  note         = {Accessed 2025-12-19}
}
```

<a id="snapfacebook2012"></a>
```bibtex
@misc{snapFacebook2012,
  author       = {{Stanford Network Analysis Project}},
  title        = {Social Circles: Facebook},
  year         = {2012},
  howpublished = {SNAP},
  url          = {https://snap.stanford.edu/data/egonets-Facebook.html},
  note         = {Accessed 2025-12-19}
}
```

<a id="yangleskovec2012groundtruth"></a>
```bibtex
@inproceedings{YangLeskovec2012GroundTruth,
  author    = {Yang, Jaewon and Leskovec, Jure},
  title     = {Defining and Evaluating Network Communities Based on Ground-Truth},
  booktitle = {Proceedings of the ACM SIGKDD Workshop on Mining Data Semantics},
  year      = {2012},
  pages     = {1--8}
}
```

## License

Licensing to be determined.

## Contact

For questions about the research or this repository, contact the corresponding author, Avin Kolahdooz, at [akolahdooz@unm.edu](mailto:akolahdooz@unm.edu).
