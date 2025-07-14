# modules/compression/model_loader.py

import os
import torch
import torch.nn as nn
import torchvision

# Local path where the Docker image clones the repo
LOCAL_HUB = "/opt/pytorch-cifar-models"
# Fallback GitHub repo if LOCAL_HUB isn't found
REMOTE_HUB = "chenyaofo/pytorch-cifar-models"

def auto_load_model(path_or_state, device):
    """
    1) If path_or_state is a filepath:
       a) Try torch.load(..., weights_only=False) → may return state_dict or pickled nn.Module
       b) If that fails, try torch.jit.load() → extract its state_dict()
    2) If it's already an nn.Module, return it.
    3) Otherwise assume it's a state_dict dict:
       • pick CIFAR-10 ResNet20/32 or VGG11_BN by filename
       • else fallback to torchvision.models.resnet18
    """
    state_dict = None

    # 1) If it's a path, try full torch.load first
    if isinstance(path_or_state, str):
        try:
            # weights_only=False to allow unpickling full nn.Module checkpoints
            ckpt = torch.load(path_or_state, map_location=device, weights_only=False)
        except Exception as e_load:
            # fallback: maybe it's TorchScript
            try:
                jm = torch.jit.load(path_or_state, map_location=device)
                ckpt = jm.state_dict()  # grab weights for a rebuild
            except Exception as e_ts:
                raise RuntimeError(
                    f"Failed to load '{path_or_state}' as checkpoint ({e_load}) "
                    f"or TorchScript ({e_ts})."
                )
    else:
        ckpt = path_or_state

    # 2) If it's already an nn.Module, just return it
    if isinstance(ckpt, nn.Module):
        return ckpt.to(device)

    # 3) Otherwise we have a raw state_dict
    state_dict = ckpt
    fn = os.path.basename(path_or_state).lower() if isinstance(path_or_state, str) else ""

    # Helper to pull CIFAR-10 models from hub
    use_local = os.path.isdir(LOCAL_HUB)
    repo = LOCAL_HUB if use_local else REMOTE_HUB
    def load_from_hub(model_name):
        kwargs = {"pretrained": False}
        if use_local:
            kwargs["source"] = "local"
        return torch.hub.load(repo, model_name, **kwargs)

    def build_and_load(base_model):
        m = base_model.to(device)
        m.load_state_dict(state_dict, strict=False)
        return m

    # 3a) CIFAR-10 special cases
    if "resnet20" in fn:
        return build_and_load(load_from_hub("cifar10_resnet20"))
    if "resnet32" in fn:
        return build_and_load(load_from_hub("cifar10_resnet32"))
    if "vgg11" in fn:
        return build_and_load(load_from_hub("cifar10_vgg11_bn"))
    if "vgg13" in fn:
        return build_and_load(load_from_hub("cifar10_vgg13_bn"))
    if "mobilenet" in fn:
        return build_and_load(load_from_hub("cifar10_mobilenetv2_x1_0"))

    # 3b) Fallback → torchvision ResNet-18
    out_feats = 10
    if isinstance(state_dict, dict) and "fc.weight" in state_dict:
        out_feats = state_dict["fc.weight"].shape[0]
    base = torchvision.models.resnet18(num_classes=out_feats)
    return build_and_load(base)
