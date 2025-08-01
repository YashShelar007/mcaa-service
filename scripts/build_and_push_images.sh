#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# now includes the two new workers
REPOS=(
  baseline
  prune_structured
  quantize
  distill_kd
  evaluate
  prune_search
  prune_and_quantize
  model_loader
)

for name in "${REPOS[@]}"; do
  REPO_NAME="mcaa-service-${name}"
  REPO_URL="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

  echo ">>> Building image for $name"

  # Ensure ECR repo exists
  if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"
  fi

  # Build for amd64, tag, and push
  docker build --platform linux/amd64 \
    -t "${REPO_NAME}:latest" \
    -f "src/workers/${name}/Dockerfile" .

  docker tag "${REPO_NAME}:latest" "${REPO_URL}:latest"

  echo ">>> Pushing ${REPO_NAME}:latest to ECR"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

  docker push "${REPO_URL}:latest"
done

echo "All images built (amd64) and pushed."
