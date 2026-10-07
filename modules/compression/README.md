# compression

The step scripts that run inside the worker containers. Each one works in two modes: from the command line on local files, or in ECS, where it reads `MODEL_BUCKET`, `MODEL_S3_KEY`, `USER_ID` and `PROFILE` (plus step-specific variables) from the environment, pulls the model from S3 and writes the result back.

| Script | Step |
|---|---|
| `baseline.py` | Measure the uploaded model |
| `prune_structured.py` | Structured channel pruning at a fixed ratio, then fine-tune |
| `prune_search.py` | Try pruning ratios 10% to 90%, keep the smallest model within the accuracy and size limits |
| `distill_kd.py` | Knowledge distillation from the baseline into the pruned model |
| `quantize.py` | Post-training static quantization (FX, with a TorchScript fallback) |
| `prune_and_quantize.py` | `prune_search` followed by `quantize` |
| `evaluate.py` | CIFAR-10 test accuracy and single-image CPU latency |
| `model_loader.py` | Loads a checkpoint, TorchScript file or state dict into a model |
| `logger.py` | Appends rows to `logs/experiment_log.csv` |

Arguments are in each script's docstring and `--help`. Dependencies: `torch`, `torchvision` (repo-root `requirements.txt`); the pruning scripts also import `torch_pruning` and the ECS mode imports `boto3`, which the worker Dockerfiles install. Not verified: local CLI runs.
