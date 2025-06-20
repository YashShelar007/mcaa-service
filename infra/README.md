# Infrastructure as Code

**Folder:** `infra/`  
**Purpose:** Defines and manages all cloud resources supporting MCaaS via Terraform.

---

## Overview

In Chapter 3 of the thesis (“System Architecture”), we propose a fully serverless, containerized pipeline orchestrated by AWS Step Functions. This directory codifies that design:

- **`api.tf`** – HTTP API (Amazon API Gateway v2) routes for `/presign`, `/submit`, `/status`, `/download` → backed by a single Lambda
- **`ecs.tf`** – ECS cluster, task definitions and services for each compression step (baseline, prune, distill, quantize, evaluate)
- **`step_functions.tf`** – State Machine definition implementing the prune → optional KD → quantize → evaluate DAG
- **`roles.tf`** – IAM roles/policies for Step Functions, Lambda, and ECS tasks with least-privilege (S3, DynamoDB, SFN)
- **`logs.tf`** – CloudWatch log groups and retention settings for all services
- **`static_site.tf`** – (Optional) S3 + CloudFront for hosting the web UI
- **`backend.tf`**, **`data.tf`**, **`providers.tf`** – Terraform backend and data sources

---

## Deployment

1. **Configure**

   - Edit `providers.tf` to set your AWS region and state backend.
   - Ensure `terraform.tfvars` contains `model_bucket`, `vpc_id`, etc., matching your account.

2. **Initialize**

   ```bash
   cd infra
   terraform init
   ```

3. **Plan & Apply**

   ```bash
   terraform plan   # review changes
   terraform apply  # provision all resources
   ```

4. **Component-specific Targets**
   If you only need to update the API or the Step Function:

   ```bash
   terraform apply -target=aws_lambda_function.api
   terraform apply -target=aws_sfn_state_machine.pipeline
   ```

5. **Teardown**

   ```bash
   terraform destroy
   ```

---

## Research Notes

- **Security** (Section 3.4): IAM policies enforce least privilege—e.g., Lambda may only call `states:StartExecution`, `GetExecutionHistory`, `s3:GetObject`, and `PutObject` on the model bucket.
- **Scalability** (Section 3.5): Fargate tasks auto-scale per invocation, avoiding cold-start for each state machine step.
- **Cost Optimization** (Section 3.6): Log retention is set to 14 days, balancing auditability vs. storage costs.

For full design rationale, see the [Thesis Proposal](../docs/MCaaS_Thesis_Proposal.pdf).
