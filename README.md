# Geometry-Enhanced Community Detection on Facebook Ego Network  
*LRC, ORC, and Hybrid LRC+ORC Curvature Models*

---

## Overview

This repository contains a full pipeline for curvature-based community detection on the **SNAP Facebook Ego Network**.  
The project evaluates and compares three methods:

1. **LRC-Only**  
   Lower Ricci Curvature is computed for every edge, low-curvature edges are pruned, and Louvain is applied.

2. **ORC-Only**  
   Ollivier–Ricci Curvature + Ricci Flow is applied to the full graph, followed by curvature-based pruning.

3. **Hybrid LRC + ORC (Proposed Method)**  
   First sparsify using LRC → then run ORC/Ricci Flow only on the reduced graph.  
   This hybrid strategy aims to:
   - reduce ORC runtime  
   - increase accuracy relative to LRC-only  
   - remain faster than full ORC  

Ground-truth labels come from `0.circles`, enabling precise ARI/NMI evaluation.

---

## 📁 Files in This Repository

- **face_book_lrc,_orc,_combined_lrc_+_orc.py**  
  Full implementation of:
  - Data loading  
  - LRC computation  
  - ORC (Sinkhorn) + Ricci Flow  
  - Sparsification  
  - Hyperparameter sweeps  
  - Final evaluation & comparison  

- **facebook_combined.txt**  
  Raw Facebook edge list from SNAP (if present).

---
Best LRC-only:   ARI=0.059, NMI=0.259, time=0.085s
ORC-only:        ARI=0.190, NMI=0.401, time=183.766s
Hybrid (best):   ARI=0.087, NMI=0.288, time=120.410s

Improvements:
+11% NMI vs LRC-only
+47% ARI vs LRC-only
34% faster than ORC-only
