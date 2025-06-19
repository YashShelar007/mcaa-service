#!/usr/bin/env python3
"""
measure_inference_time.py
A utility to measure average inference time for a PyTorch model on a specified device.
"""

import time
import torch
import torch.nn as nn

def measure_inference_time(model, device, input_shape=(1, 3, 224, 224), 
                           num_warmup=10, num_runs=50):
    """
    Measures the average forward pass time for a model on given device.
    :param model: PyTorch model (already loaded with state_dict).
    :param device: torch.device (e.g. "cpu", "cuda", "mps").
    :param input_shape: shape of the input tensor (batch, channels, height, width).
    :param num_warmup: number of warm-up iterations (not timed).
    :param num_runs: number of timed iterations.
    :return: average time in seconds per forward pass.
    """
    model.eval()
    model.to(device)

    # Create a dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Warm-up
    for _ in range(num_warmup):
        _ = model(dummy_input)

    # Timed runs
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time

if __name__ == "__main__":
    # Example usage
    import sys
    from torchvision.models import resnet18

    if len(sys.argv) < 3:
        print("Usage: python measure_inference_time.py <model_path> <device>")
        print("Example: python measure_inference_time.py baseline_resnet18_cifar.pth cpu")
        sys.exit(1)

    model_path = sys.argv[1]
    device_str = sys.argv[2]

    device = torch.device(device_str)

    # Example: load a ResNet18 with 10 classes
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))

    avg_sec = measure_inference_time(model, device, input_shape=(1,3,32,32))
    print(f"Average inference time for {model_path} on {device_str}: {avg_sec*1000:.3f} ms")
