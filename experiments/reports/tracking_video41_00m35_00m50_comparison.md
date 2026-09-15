# Video 41 tracking experiment

Clip: `00:35-00:50` from `video41`

The two runs use the same model, clip, IoU threshold (`0.5`), and target CSV.
Only the detector confidence and ByteTrack configuration changed.

| Metric | Baseline | ByteTrack experiment |
|---|---:|---:|
| Confidence | 0.25 | 0.15 |
| Track buffer | 30 (default) | 60 |
| High threshold | 0.25 (default) | 0.15 |
| Low threshold | 0.10 (default) | 0.05 |
| New track threshold | 0.25 (default) | 0.15 |
| Match threshold | 0.80 (default) | 0.85 |
| Predicted objects | 212 | 221 |
| Precision | 0.9198 | 0.8597 |
| Recall | 0.6250 | 0.6090 |
| F1 | 0.7443 | 0.7129 |
| MOTA | 0.3429 | 0.3013 |
| Mean IoU | 0.7136 | 0.6982 |
| ID switches | 71 | 65 |

## Interpretation

The experiment produced slightly fewer ID switches, but the lower detection
threshold introduced more false positives and did not improve recall on this
clip. The baseline remains the better overall configuration for this sequence.
The experimental configuration is retained for reproducibility in
`configs/bytetrack_video41_experiment.yaml`.
