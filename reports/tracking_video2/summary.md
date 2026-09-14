# Tracking report

## Video

| Metric | Value |
|---|---:|
| Duration | 25.0 s |
| Source FPS | 30.0 |
| Processed frames | 750 |
| Detection coverage | 100.0% |
| Mean detections/frame | 1.624 |
| Mean confidence | 0.7139 |
| Inference FPS | 4.99 |
| Inference latency p50 | 182.86 ms |
| Inference latency p95 | 378.18 ms |
| Realtime at source FPS | False |

## Tracking proxies

| Metric | Value |
|---|---:|
| Unique track IDs | 34 |
| Longest track | 290 frames |
| Tracks seen once | 9 |
| Reappearance gaps | 62 |
| Largest gap | 29 frames |

## Classes

| Class | Detection count |
|---|---:|
| Clipper | 2 |
| Grasper | 1166 |
| Hook | 50 |

## Important limitation

MOTA, IDF1, and HOTA are not computed because this video has no ground-truth
trajectories. The values above measure system behavior, not tracking accuracy.
See `frame_metrics.csv` for frame-level details.
See `track_metrics.csv` for one row per observed track ID.
