#!/usr/bin/env python3
"""
quantize.py

Post-training static 8-bit quantization.

Dual mode:
 1) CLI: python quantize.py <in_model> <out_model> [--bitwidth B]
 2) ECS: runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE, BITWIDTH env-vars
"""
import os, sys, time, warnings, io, json
import torch, torch.nn as nn, torch.quantization as quant
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import quantize_jit
from torch.ao.quantization import QConfigMapping
from torchvision import transforms, datasets
from logger import log
from model_loader import auto_load_model

# CIFAR-10 normalization
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)

def get_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_C10_MEAN, _C10_STD),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            _, pred = model(imgs).max(1)
            correct += (pred == labs).sum().item()
            total += labs.size(0)
    return 100.0 * correct / total

def measure_latency(model, device, input_shape=(1,3,32,32), warmup=10, runs=50):
    model.eval().to(device)
    dummy = torch.randn(*input_shape).to(device)
    # warm-up
    for _ in range(warmup):
        _ = model(dummy)
    # timed runs
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    return (time.time() - start) / runs * 1000.0  # ms

def script_and_save(model, path):
    """JIT-script & save everything into a single .pt"""
    try:
        jm = torch.jit.script(model)
    except Exception:
        jm = torch.jit.trace(model, torch.randn(1,3,32,32))
    torch.jit.save(jm, path)

def _choose_qengine():
    """Prefer FBGEMM on x86, else QNNPACK."""
    engines = torch.backends.quantized.supported_engines
    if 'fbgemm' in engines:
        return 'fbgemm'
    if 'qnnpack' in engines:
        return 'qnnpack'
    raise RuntimeError(f"No quant engine available; supported={engines}")

# ─── FX‐Quant helper ────────────────────────────────────────────────────
def apply_fx_static_quant(model, calib_loader):
    # 1) pick the best engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine

    # 2) build a config dict: "": default qconfig
    qconfig = quant.get_default_qconfig(engine)
    mapping = QConfigMapping()
    mapping.set_global(qconfig)

    # 3) symbolically trace + insert observers
    # 3) we must also pass an example input batch for tracing
    model.eval().to("cpu")
    # grab one batch of images (no labels needed)
    example_inputs, _ = next(iter(calib_loader))
    # make sure it's on CPU
    example_inputs = example_inputs.to("cpu")
    # now prepare_fx can trace with that tensor
    prepared = prepare_fx(model, mapping, example_inputs)

    # 4) calibrate on ~20 batches
    with torch.no_grad():
        for i, (imgs, _) in enumerate(calib_loader):
            if i >= 20: break
            prepared(imgs)

    # 5) convert to int8 graph
    return convert_fx(prepared)

def apply_jit_quant(scripted_mod, calib_loader, backend="qnnpack"):
    """
    JIT‐quantize a ScriptModule in‐place using QNNPACK.
    Needs a small batch to run through for calibration.
    """
    # 1) pick engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine

    # 2) build QConfigMapping
    # qconfig = quant.get_default_qconfig(backend)
    # mapping = QConfigMapping().set_global(qconfig)
    # 2) build a simple qconfig_dict for static quant
    qconfig = quant.get_default_qconfig(engine)
    qconfig_dict = {"": qconfig}

    # 3) grab one batch for calibration
    scripted_mod = scripted_mod.to("cpu").eval()
    example_inputs, _ = next(iter(calib_loader))
    example_inputs = example_inputs.to("cpu")

    # 4) define run function & args for calibration
    run_fn   = lambda m, x: m(x)
    run_args = (example_inputs,)

    # 5) quantize via JIT
    return quantize_jit(scripted_mod, qconfig_dict, run_fn, run_args)

# ── CLI Mode ────────────────────────────────────────────────────────────────
def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("in_model",  help="Path to model (.pth or .pt)")
    p.add_argument("out_model", help="Where to save quantized .pt")
    p.add_argument("--bitwidth","-b", type=int, default=8,
                   help="8-bit static quant; else FP32")
    args = p.parse_args()

    device   = torch.device("cpu")
    loader   = get_test_loader()
    model_fp = auto_load_model(args.in_model, device)

    # BEFORE
    sz0  = os.path.getsize(args.in_model)/1024**2
    acc0 = evaluate(model_fp, loader, device)
    lat0 = measure_latency(model_fp, device)
    print(f"[CLI] BEFORE quant → size={sz0:.2f}MiB, acc={acc0:.2f}%, lat={lat0:.2f}ms")

    # QUANTIZE or just re-save
    if args.bitwidth == 8:
        if isinstance(model_fp, torch.jit.RecursiveScriptModule):
            print(f"[CLI] JIT-quantizing scripted model on {_choose_qengine()}…")
            model_q = apply_jit_quant(model_fp, loader)
        else:
            print(f"[CLI] FX Static 8-bit quantization on {_choose_qengine()}…")
            model_q = apply_fx_static_quant(model_fp, loader)
    else:
        print(f"[CLI] Skipping Quant (bitwidth={args.bitwidth}), re-save FP32…")
        model_q = model_fp

    script_and_save(model_q, args.out_model)

    # AFTER
    sz1  = os.path.getsize(args.out_model)/1024**2
    acc1 = evaluate(model_q, loader, device)
    lat1 = measure_latency(model_q, device)
    print(f"[CLI] AFTER  quant → size={sz1:.2f}MiB, acc={acc1:.2f}%, lat={lat1:.2f}ms")
    print(json.dumps({"accuracy":acc1, "size_mb":int(sz1*1024**2)}))

# ── ECS Mode ────────────────────────────────────────────────────────────────
def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket  = os.environ["MODEL_BUCKET"]
    key     = os.environ["MODEL_S3_KEY"]
    user    = os.environ["USER_ID"]
    profile = os.environ["PROFILE"]
    bitwidth = int(os.environ.get("BITWIDTH", 8))

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    basename = os.path.basename(key)
    tmp_in   = f"/tmp/{basename}"
    tmp_out  = f"/tmp/quant_{basename}"

    print(f"[ECS] Downloading s3://{bucket}/{key} → {tmp_in}")
    s3.download_file(bucket, key, tmp_in)

    device   = torch.device("cpu")
    loader   = get_test_loader()
    model_fp = auto_load_model(tmp_in, device)

    # BEFORE
    sz0  = os.path.getsize(tmp_in)/1024**2
    acc0 = evaluate(model_fp, loader, device)
    lat0 = measure_latency(model_fp, device)
    print(f"[ECS] BEFORE quant → size={sz0:.2f}MiB, acc={acc0:.2f}%, lat={lat0:.2f}ms")

    # QUANTIZE or just re-save
    if bitwidth == 8:
        if isinstance(model_fp, torch.jit.RecursiveScriptModule):
            print(f"[ECS] JIT-quantizing scripted model on {_choose_qengine()}…")
            model_q = apply_jit_quant(model_fp, loader)
        else:
            print(f"[ECS] FX Static 8-bit quantization on {_choose_qengine()}…")
            model_q = apply_fx_static_quant(model_fp, loader)
    else:
        print(f"[CLI] Skipping Quant (bitwidth={bitwidth}), re-save FP32…")
        model_q = model_fp

    # save & log
    script_and_save(model_q, tmp_out)

    sz1  = os.path.getsize(tmp_out)/1024**2
    acc1 = evaluate(model_q, loader, device)
    lat1 = measure_latency(model_q, device)

    print(f"[ECS] AFTER  quant → size={sz1:.2f}MiB, acc={acc1:.2f}%, lat={lat1:.2f}ms")

    # UPLOAD & METADATA
    out_key = f"users/{user}/{profile}/quantized/{basename}"
    s3.upload_file(tmp_out, bucket, out_key)
    print(f"[ECS] Uploaded quantized model to s3://{bucket}/{out_key}")
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "quantize"},
        "Accuracy":  {"N": f"{acc1:.2f}"},
        "SizeMB":    {"N": f"{sz1:.3f}"},
        "LatencyMS": {"N": f"{lat1:.2f}"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":    "quantize",
      "step":      "quantize",
      "accuracy":  acc1,
      "size_mb":   sz1,
      "latency_ms": lat1,
    })

    print(json.dumps({"accuracy":acc1, "size_mb":int(sz1*1024**2)}))
    print("[ECS] Quantization step complete.")

if __name__=="__main__":
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
        cli_mode()
    else:
        ecs_mode()
