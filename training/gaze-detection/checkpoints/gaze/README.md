# Learned gaze model ("visual model")

ETH-XGaze ResNet-18 fine-tuned on all 15 MPIIFaceGaze people with
low-resolution + JPEG augmentation: the `final_all` run of
`notebooks/gaze_transfer_learning.ipynb`. Its held-out accuracy comes from the
same configuration's leave-one-person-out run, **4.54°** (MediaPipe iris
offsets: 7.18°), in `../../results/gaze_model_results.json`.

Rebuilt with `kaggle_gaze_deploy.py` (Kaggle notebook "OpenMeet gaze model
rebuild", GPU T4 x2, 16 min), because the original session's files were not
kept. The `final_all` run has no test set, so the rebuild is checked by its
training log and the ONNX parity check, not re-scored.

| File | What |
|---|---|
| `openmeet_gaze_resnet18.onnx` | Input `face`: normalised face crop, uint8 RGB `(N, 224, 224, 3)` (Zhang 2018 normalisation, notebook cell 3); output `pitch_yaw` in radians. ImageNet normalisation is inside the graph. 44.7 MB, 34 ms/face on 1 CPU thread. |
| `model.pt` | PyTorch `state_dict` for `timm.create_model('resnet18', num_classes=2)`. |
| `cfg.json`, `train_log.txt`, `onnx_info.json` | Training configuration, per-epoch loss, ONNX parity (max diff 2.4e-7 rad) and CPU timing. |

Not used by the sidecar yet: live frames first need face normalisation from
MediaPipe head pose (see `../../README.md`, "Learned gaze model").

**License: CC BY-NC-SA 4.0, non-commercial use only.** These weights are
derived from MPIIFaceGaze (Zhang et al. 2017, *It's written all over your
face*) and the ETH-XGaze ResNet-18 (Zhang et al. 2020, *ETH-XGaze*), both
CC BY-NC-SA 4.0. Cite both and share any derivative under the same license.
