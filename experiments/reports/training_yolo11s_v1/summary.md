# YOLO11s training summary

## Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11s pretrained |
| Dataset | Cholec80, `/content/Cholec80-3/data.yaml` |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 px |
| Seed | 42 |
| Validation split | `val` |
| Classes | Bipolar, Clipper, Grasper, Hook, Irrigator, Scissors, Specimen Bag |

## Best validation result

The best `mAP50-95` occurred at epoch 49.

| Metric | Value |
|---|---:|
| Precision | 0.90924 |
| Recall | 0.88441 |
| mAP50 | 0.92569 |
| mAP50-95 | 0.55166 |

The final epoch 50 was slightly lower: mAP50 `0.92236` and mAP50-95
`0.55029`. The repository uses the exported `best.pt`, not `last.pt`.

## Interpretation

The detector is a useful baseline, but these are validation metrics only. A
separate test split and per-class metrics should be added before describing
the results as generalization performance. The normalized confusion matrix
also indicates that class imbalance and confusion involving Grasper and
Specimen Bag deserve a short failure analysis.

The ZIP archive contains the source plots, `results.csv`, `args.yaml`, and
`weights/best.pt`. Its `best.pt` has the same SHA-256 hash as the current local
`models/best.pt`, so no model replacement was necessary.