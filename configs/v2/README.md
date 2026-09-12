# V2 experiment configuration

`model_scaling.yaml` is the experiment contract for the controlled ESM-2
8M/35M/150M comparison. V1 outputs remain unchanged.

The existing train/validation/test assignments are preserved for direct V1
comparison. Model and representation selection uses validation data only. The
already-inspected test split is reserved for the final selected configuration
and must continue to be described as exploratory.

Large embeddings, fitted models, and PyTorch checkpoints remain outside Git.
Compact run configurations and aggregate metrics may be committed.

