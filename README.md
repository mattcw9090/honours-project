# Honours Project — Reproducible Modelling & Validation on Noisy Real-World Data (Python)

This repository contains my honours project work focused on building **reproducible analysis + modelling pipelines** with clear evaluation and documentation. While the applied domain here is computer vision (maritime object detection), the workflow generalises to many data science settings: working with messy real-world data, quantifying uncertainty, and communicating evidence-backed tradeoffs.

---

## Project summary

**Goal:** Build a transparent, in-domain benchmark for maritime object detection on the **ABOShips** dataset (9,880 images, 11 classes).  
**Models compared:** YOLO11 / YOLO12 (one-stage) vs RT-DETRv2 / RF-DETR (transformer-based).  
**Protocol:** fixed 80/10/10 split, 640×640 letterbox, shared post-processing, batch-1 runtime harness (no TTA).  
**Outputs:** COCO AP metrics, PR curves, F1 vs confidence sweeps, plus deployment metrics (latency/FPS/peak memory).

---

## Key results (ABOShips @ 640×640, batch=1, no TTA)

| Model      | AP50  | AP75  | AP50–95 | Latency (ms) | FPS  | Peak GPU Mem (MB) |
|-----------|-------|-------|---------|--------------|------|-------------------|
| YOLO11    | 0.613 | 0.385 | 0.313   | 2.7          | 370  | 1150              |
| YOLO12    | 0.610 | 0.377 | 0.311   | 3.6          | 280  | 1220              |
| RT-DETRv2 | 0.591 | 0.349 | 0.287   | 6.6          | 150  | 1940              |
| RF-DETR   | 0.666 | 0.421 | 0.339   | 4.4          | 227  | 3120              |

**Main failure mode:** small, distant targets near the horizon (a good case study in rare-event detection and class imbalance).

> Note: timings are hardware-dependent. These figures were measured under a consistent GPU runtime harness.

---

## How to reproduce (Colab-friendly)

### 1) Prepare datasets (YOLO + COCO layouts)
1. Run `notebooks/YOLO Data Prep.ipynb`  
   - Creates YOLO folder structure and writes `dataset.yaml`
2. Run `notebooks/ABOShips to COCO.ipynb`  
   - Exports COCO JSON per split
   - Clips boxes to image bounds, filters invalid annotations

### 2) Train / evaluate models
- `notebooks/YOLO11.ipynb`
- `notebooks/YOLO12.ipynb`
- `notebooks/RT-DETRv2.ipynb`
- `notebooks/RF-DETR.ipynb`

### 3) Evaluation protocol (fixed for comparability)
- 640×640 letterbox inputs
- batch size = 1 for timing
- no test-time augmentation
- consistent post-processing rules across runs

---

## Data and thesis availability

- The **ABOShips dataset is not included** in this repository and is shared according to its original license.  
- The **honours thesis (PDF)** and any additional supporting materials are **available upon request**.  
- This repository contains only the **code, configuration files, and notebooks** required to reproduce the analysis using appropriately sourced data.

This structure reflects standard practice when working with licensed, private, or sensitive datasets.

