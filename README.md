# mcaa-service

Model compression as a service: upload a PyTorch CIFAR-10 image classifier, pick a compression profile or set your own limits, and an AWS Step Functions pipeline prunes, distills and quantizes it on Fargate and hands back a smaller model with size and accuracy figures. It started as a master's thesis project at Arizona State University (advisor Dr. Ming Zhao); the proposal is in `docs/MCaaS_Thesis_Proposal.pdf`.

## What it does not do

- Only CIFAR-10 models. Evaluation and fine-tuning load CIFAR-10 data, and the model loader knows five CIFAR-10 architectures (ResNet20, ResNet32, VGG11_BN, VGG13_BN, MobileNetV2) plus a ResNet-18 fallback.
- No CLI. `src/cli` was never built.
- No tests. `tests/` was never filled in.
- The Terraform has hard-coded names (S3 buckets `mcaa-service-model-storage`, `mcaa-service-ui` and the state bucket `mcaa-service-tf-state`, region `us-west-2`), so it cannot be applied as is by anyone else.
- No results are published here. Not documented: accuracy or size reductions per profile.

## Quickstart

Not verified end to end: the pipeline needs an AWS account, the hard-coded bucket names changed to your own, and Docker. What was checked on 2026-10-07: every Python file compiles, and `terraform init -backend=false && terraform validate` passes (with a deprecation warning for `aws_subnet_ids`).

```bash
# 1. Package the API Lambda (infra/lambda/api.zip is not committed)
./infra/lambda/build.sh

# 2. Provision (edit bucket names in infra/*.tf and the state bucket in infra/backend.tf first)
cd infra && terraform init && terraform apply && cd ..

# 3. Build and push the eight worker images to ECR
./scripts/build_and_push_images.sh

# 4. Point src/ui/index.html at your API (API_BASE) and publish the UI
./scripts/deploy_ui.sh
```

The compression scripts also run on their own against a local checkpoint, given `torch` and `torchvision` (`pip install -r requirements.txt`). For example `python modules/compression/quantize.py in.pth out.pt --bitwidth 8`. Each script's docstring lists its arguments. Not verified: these CLI runs.

## How it works

The browser UI asks the API Lambda for a presigned S3 POST, uploads the model straight to S3, then calls `/submit`. The Lambda validates the options (`acc_tol`, `size_limit`, `bitwidth`) and starts the Step Functions state machine. The UI polls `/status`, which reads the execution history and returns accuracy and size from each finished task, and finally fetches a presigned download URL from `/download`.

The state machine runs one Fargate task per step. A Choice state picks the route:

- Only a bit width given: quantize the baseline.
- Accuracy tolerance or size limit given, no bit width: `prune_search` tries pruning at 10% steps up to 90%, fine-tunes each candidate, and keeps the smallest model that meets the limits.
- Both: `prune_and_quantize` runs the search, then quantizes.
- Otherwise the preset profile decides: `balanced` prunes, distills (knowledge distillation from the baseline), then quantizes; `high_accuracy` distills only; anything else (the UI's `max_compression`) quantizes the baseline only.

Every route ends with an `evaluate` task. Models land under `users/<user_id>/<profile>/<step>/` in the models bucket. Workers also write a row to a DynamoDB table and to `logs/experiment_log.csv`.

```
modules/compression/   baseline, prune_structured, prune_search, distill_kd,
                       quantize, prune_and_quantize, evaluate, model_loader, logger
src/workers/           one Dockerfile per step (4 vCPU / 8 GB Fargate tasks)
src/ui/index.html      single-page UI, Tailwind from a CDN, vanilla JS
infra/                 Terraform: API Gateway + Lambda, ECS, Step Functions, S3, CloudFront, IAM
infra/lambda/api/      Lambda handler: presign, submit, status, download
models/                demo CIFAR-10 checkpoints (about 90 MB)
```

## Known limits

- The Lambda sends `Access-Control-Allow-Origin: *` and has no authentication; `user_id` is whatever the caller passes.
- `models/` holds binary checkpoints in git, about 90 MB.
- `src/ui/index.html` contains the API Gateway URL of the original deployment.
- `src/workers/prune_search/Dockerfile` does not install `torch-pruning`, while `prune_search.py` imports `prune_structured`, which does. Whether that image runs was not checked.
- `aws_subnet_ids` is deprecated in current AWS provider versions.

## Status

Built in 2025 as a cloud computing and thesis project. Archived; not maintained.

## License

MIT. See `LICENSE`.
