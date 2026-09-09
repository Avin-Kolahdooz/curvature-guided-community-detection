# Curvature-Guided Community Detection

A reproducible framework for graph community detection using **Lower Ricci Curvature (LRC)** and **Ollivier–Ricci Curvature (ORC)** to guide graph sparsification and improve the efficiency of downstream community detection.

This repository contains the implementation, experiments, and reproducibility resources associated with our work on curvature-guided community detection.

---

## Overview

Community detection is a fundamental problem in network science, with applications in social networks, biological systems, information networks, and other complex graph-structured data.

This project investigates whether geometric information derived from discrete Ricci curvature can be used to identify structurally important edges before community detection.

We study two complementary curvature measures:

- **Lower Ricci Curvature (LRC)** — a computationally efficient, locally defined curvature measure used for early-stage graph sparsification.
- **Ollivier–Ricci Curvature (ORC)** — an optimal-transport-based curvature measure that provides richer geometric information but is substantially more computationally expensive.

The main framework combines these methods in a two-stage pipeline:

**Graph → LRC computation → LRC-guided pruning → ORC / Ricci flow → ORC-guided pruning → Community detection → Evaluation**

The central objective is to study the trade-off between **computational efficiency**, **graph sparsification**, and **community recovery quality**.

---

## Key Features

- Two-stage **LRC → ORC** curvature-guided sparsification pipeline
- Ollivier–Ricci curvature and Ricci flow for graph reweighting
- Louvain-based community detection
- Synthetic benchmark experiments with known ground-truth communities
- Experiments on multiple real-world network datasets
- Joint LRC–ORC pruning sensitivity analysis
- Evaluation using **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**
- Runtime analysis across different pruning regimes
- Reproducible experiment notebooks and shared utility functions

---

## Methodology

The proposed workflow consists of four main stages.

### 1. Lower Ricci Curvature

LRC provides a computationally efficient approximation of local geometric structure.

Edges with low curvature are candidates for early pruning, allowing the graph to be reduced before applying the more computationally expensive ORC calculation.

### 2. Ollivier–Ricci Curvature and Ricci Flow

ORC measures the geometric relationship between neighboring probability distributions using optimal transport.

Ricci flow iteratively updates edge weights using curvature information, allowing the graph geometry to evolve before the second pruning stage.

### 3. Curvature-Guided Sparsification

The framework investigates different combinations of LRC and ORC pruning ratios to determine how graph reduction affects both computational cost and community recovery.

Joint sensitivity experiments evaluate multiple combinations of LRC and ORC pruning levels.

### 4. Community Detection and Evaluation

Community structure is recovered using the Louvain algorithm.

When ground-truth labels are available, performance is evaluated using:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Runtime is also recorded to quantify the computational trade-off introduced by different sparsification strategies.

---

## Experimental Datasets

The framework is evaluated on both synthetic and real-world networks.

### Synthetic Benchmark

Synthetic networks provide controlled community structure and known ground-truth labels, allowing direct evaluation of community recovery under different curvature and pruning configurations.

### Real-World Networks

Experiments are included for:

- **Facebook** — social network
- **Amazon** — product co-purchasing network
- **DBLP** — academic collaboration network
- **YouTube** — social network

For each real-world benchmark, empirical within-community and between-community edge densities are used to characterize the strength of the observed community structure.

---

## Repository Structure

```text
curvature-guided-community-detection/
│
├── src/
│   └── community_detection_utils.py
│
├── notebooks/
│   ├── Synthetic_Module_Verification.ipynb
│   ├── Facebook.ipynb
│   ├── Amazon.ipynb
│   ├── DBLP.ipynb
│   └── Youtube.ipynb
│
├── data/
│   └── README.md
│
├── results/
│   └── ...
│
├── figures/
│   └── ...
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
