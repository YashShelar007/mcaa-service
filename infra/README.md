# infra

Terraform for the AWS side, region `us-west-2`. State lives in the S3 bucket and DynamoDB lock table named in `backend.tf`, which must exist before `terraform init`.

- `api.tf`: HTTP API (API Gateway v2) with routes `/presign`, `/submit`, `/status`, `/download`, all handled by one Lambda built from `lambda/api` (run `lambda/build.sh` first; `api.zip` is not committed).
- `ecs.tf`: ECS cluster, an ECR repository and a Fargate task definition per worker.
- `step_functions.tf`: the state machine. Routes are described in the top-level README.
- `main.tf`: models bucket (versioned, encrypted, no public access) and the DynamoDB metadata table.
- `static_site.tf`: UI bucket behind CloudFront.
- `roles.tf`: IAM roles for the tasks, Step Functions and the Lambda.
- `logs.tf`: CloudWatch log groups, 14-day retention.

Bucket and table names are literals in the files, not variables. Change them before applying in your own account.

```bash
terraform init
terraform plan
terraform apply
```

Not verified: apply. `terraform validate` passes.
