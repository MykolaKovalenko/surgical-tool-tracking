# YOLO11s v2 training and comparison summary

## Configuration

| Parameter | v2 |
|---|---|
| Starting model | YOLO11s `last.pt` from the previous run |
| Dataset | Cholec80, validation split `val` |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 px |
| Seed | 0 |
| Additional augmentation | `degrees=15.0` |

## Best validation result

The best `mAP50-95` occurred at epoch 88.

| Metric | v1 | v2 |
|---|---:|---:|
| Precision | 0.90924 | 0.93707 |
| Recall | 0.88441 | 0.89118 |
| mAP50 | 0.92569 | 0.92980 |
| mAP50-95 | 0.55166 | 0.55009 |

The v2 model improves precision and mAP50 slightly, but its mAP50-95 is
slightly lower. Both models use the same seven classes and validation split.

## Same-video tracking benchmark

Both models were evaluated on `data/test_video2.mp4` with confidence `0.25`,
image size `640`, CPU inference, and `bytetrack.yaml`.

| Diagnostic | v1 | v2 |
|---|---:|---:|
| Inference FPS | 10.62 | 10.76 |
| Latency p50 | 92.18 ms | 90.98 ms |
| Latency p95 | 98.84 ms | 96.36 ms |
| Detection coverage | 100.00% | 99.33% |
| Unique track IDs | 34 | 37 |
| Tracks seen once | 9 | 5 |
| Reappearance gaps | 62 | 92 |
| Largest gap | 29 frames | 22 frames |

## Interpretation

v2 is not unambiguously better for tracking. It produces fewer one-frame
tracks and shorter largest gaps, but more reappearance gaps and more unique
IDs. Ground-truth identities are still required for IDF1, HOTA, MOTA, ID
switches, and fragmentation. The correct next step is a short annotated
tracking sequence, not selecting a model from mAP alone.