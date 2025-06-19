#!/usr/bin/env bash
set -euo pipefail

# Variables
BUCKET_NAME="mcaa-service-ui"
BUILD_DIR="src/ui"
REGION="us-west-2"
INFRA_DIR="infra"

echo "Syncing UI files to S3 bucket: $BUCKET_NAME"
aws s3 sync \
  "$BUILD_DIR" \
  "s3://$BUCKET_NAME" \
  --region "$REGION" \
  --delete \
  --acl private

echo "Reading CloudFront distribution ID from Terraform output..."
# Change into infra to get the output
pushd "$INFRA_DIR" >/dev/null
TF_DIST_ID=$(terraform output -raw ui_distribution_id)
popd >/dev/null

if [[ -n "$TF_DIST_ID" ]]; then
  echo "Invalidating CloudFront distribution: $TF_DIST_ID"
  aws cloudfront create-invalidation \
    --distribution-id "$TF_DIST_ID" \
    --paths "/*"
  echo "Invalidation created."
else
  echo "Warning: No CloudFront distribution ID found. Skipping invalidation."
fi

echo "UI deployment complete!"
