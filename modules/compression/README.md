# Compression Modules

**Folder:** `modules/compression/`  
**Purpose:** Implements core algorithms—pruning, knowledge distillation, dynamic quantization, and latency evaluation.

---

## Overview

Chapter 4 (“Compression Techniques”) demonstrates how MCaaS applies:

1. **Pruning** (`prune_structured.py`)

   - Magnitude-based neuron/weight pruning
   - Sparsity schedules configured via `--amount` argument

2. **Knowledge Distillation** (`distill.py`)

   - Student ← Teacher training with temperature‐scaled soft targets
   - Configurable α (teacher vs. ground‐truth mix)

3. **Dynamic Quantization** (`quantize_dynamic.py`)

   - CPU dynamic quantization on `nn.Linear` layers
   - CLI and ECS modes via environment variables

4. **Latency & Size Measurement** (`measure_inference_time.py`)

   - Warm-up & timed dummy inputs for CPU latency
   - Reports model file size in MB

5. **Logging Utility** (`logger.py`)
   - Appends per-step metadata (size, latency, profile) to DynamoDB and/or CSV

---

## Installation

Create a virtual environment and install dependencies:

```bash
cd modules/compression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### CLI Mode

```bash
# Prune
python prune_structured.py --input-model demo_resnet18.pth --output-model pruned.pth --amount 0.5

# Distill
python distill.py --teacher baseline.pth --student pruned.pth --output distilled.pth

# Quantize
python quantize_dynamic.py pruned.pth quantized.pth

# Evaluate
python measure_inference_time.py
```

### ECS Mode

When run inside Docker/ECS, each script reads:

- `MODEL_BUCKET`, `MODEL_S3_KEY`, `USER_ID`, `PROFILE` from env
- Uploads results back to `s3://<MODEL_BUCKET>/users/{USER_ID}/{PROFILE}/{step}/...`

---

## Testing

- **Unit Tests** (planned): Compare output file sizes before/after pruning & quantization.
- **Integration Tests**: Run the full pipeline on `demo_model_valid.pt` and verify DynamoDB entries.

---

## Research Notes

- **Choice of Pruning Algorithm** (Section 4.2): We adopt global magnitude pruning for simplicity; future work could integrate structured methods.
- **Distillation Loss** (Section 4.3): Temperature and α values were tuned empirically on CIFAR-10 (see Appendix A).
- **Quantization Strategy** (Section 4.4): Dynamic quantization trades off minimal accuracy loss vs. runtime speedup on CPU.
