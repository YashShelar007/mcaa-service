#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOS=(baseline prune quantize distill evaluator)

for name in "${REPOS[@]}"; do
  REPO_NAME="mcaa-service-${name}"
  REPO_URL="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

  echo ">>> Building image for $name"
  # Ensure ECR repo exists
  if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"
  fi

  # Build & tag
  docker build -t "${REPO_NAME}:latest" -f "src/workers/${name}/Dockerfile" .
  docker tag "${REPO_NAME}:latest" "${REPO_URL}:latest"

  # Push
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  docker push "${REPO_URL}:latest"
done

echo "All images built and pushed."
