# MCaaS: Model Compression as a Service

_Master’s Thesis Project_  
**Author:** Yash Shelar  
**Advisor:** Dr. Ming Zhao, SCAI, Arizona State University
**Proposal:** [MCaaS Thesis Proposal (PDF)](docs/MCaaS_Thesis_Proposal.pdf)

---

## Abstract

Deep learning models have grown increasingly large, making deployment on resource-constrained devices challenging. **MCaaS** (Model Compression as a Service) provides an end-to-end, serverless pipeline for pruning, quantization, and knowledge-distillation of PyTorch models. Users upload a pretrained model, select a “compression profile,” and receive a compressed model optimized for their latency, size, or accuracy requirements.

---

## Key Contributions

1. **Serverless Pipeline Architecture**
   - AWS Step Functions orchestration
   - Fargate-based ECS tasks for each compression step
2. **Flexible Compression Profiles**
   - _Balanced_: prune → distill → quantize
   - _High Accuracy_: distill first
   - _Max Compression_: skip distill
3. **Single-Page Web UI & CLI**
   - Intuitive upload, progress tracking, and download
   - Fully asynchronous, no page reloads
4. **Comprehensive Evaluation**
   - Automated latency, size, (optionally accuracy) measurement
   - DynamoDB metadata for analysis

---

## Project Structure

```

├── docs/                      ← Thesis proposal and documentation
│   └── MCaaS\_Thesis\_Proposal.pdf
├── infra/                     ← Terraform infra (API, ECS, Step Functions)
├── modules/compression/       ← Python modules: prune, distill, quantize, evaluate
├── src/
│   ├── api/                   ← Lambda/API Gateway code
│   ├── ui/                    ← Single-page web app (Tailwind + vanilla JS)
│   ├── cli/                   ← Command-line client (TBD)
│   └── workers/               ← Dockerfiles for ECS tasks
├── tests/                     ← Unit/integration tests (TBD)
├── models/                    ← Demo models for testing
└── README.md                  ← This file

```

---

## Quick Start

1. **Provision infrastructure**
   ```bash
   cd infra
   terraform init
   terraform apply
   ```

````

2. **Build & deploy Lambdas**

   ```bash
   cd infra/lambda
   ./build.sh
   terraform apply -target=aws_lambda_function.api
   ```
3. **Build & push ECS containers**

   ```bash
   ./scripts/build_and_push_images.sh
   ```
4. **Deploy UI**

   ```bash
   cd src/ui
   aws s3 sync . s3://<your-static-site-bucket>/
   ```
5. **Use the web app**
   Open the CloudFront/S3 URL, upload your `.pth` model, choose a profile, and watch the pipeline.

````
