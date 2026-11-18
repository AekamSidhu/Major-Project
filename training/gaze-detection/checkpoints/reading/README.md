# Reading detector checkpoints

PyTorch `state_dict`s of the chosen reading network (`readgaze/seqmodel.py`,
1-D CNN on 10 s gaze windows at 10 fps), one per training seed, from the
Kaggle run written up in `../../README.md` ("Reading detector").

`local-ai/integrity/models/reading_network.onnx` is built from these three
files (the seeds' probabilities averaged inside one graph):

    python export_onnx.py checkpoints/reading/model_cnn_gpu_seed0.pt \
        checkpoints/reading/model_cnn_gpu_seed1.pt checkpoints/reading/model_cnn_gpu_seed2.pt \
        --results results/reading_detector_results.json --out ../../local-ai/integrity/models

Re-exporting them gives outputs identical to the shipped ONNX (max difference 0.0).
