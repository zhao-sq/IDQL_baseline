from typing import Callable, Optional, Sequence
import flax.linen as nn
from flax.core import freeze, unfreeze
import jax.numpy as jnp
import flax

def _torch_conv_to_flax(w):
    # PyTorch [out_c, in_c, kH, kW] -> Flax [kH, kW, in_c, out_c]
    return jnp.asarray(w.detach().cpu().numpy().transpose(2, 3, 1, 0))


def _torch_dense_to_flax(w):
    # PyTorch [out_dim, in_dim] -> Flax [in_dim, out_dim]
    return jnp.asarray(w.detach().cpu().numpy().transpose(1, 0))


def _torch_vec(w):
    return jnp.asarray(w.detach().cpu().numpy())


def _extract_state_dict(ckpt):
    # 常见几种格式兜底
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def _strip_prefix_if_present(state_dict, prefixes):
    keys = list(state_dict.keys())
    for p in prefixes:
        matched = [k for k in keys if k.startswith(p)]
        if matched:
            return {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}
    return state_dict


def load_r3m_resnet18_into_flax(
    variables,
    torch_ckpt_path: str,
):
    """
    variables: Flax variables from model.init(...)
    torch_ckpt_path: 本地 R3M pytorch ckpt 路径
    返回更新后的 variables
    """
    import torch

    ckpt = torch.load(torch_ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    # 兼容一些常见包装前缀
    state_dict = _strip_prefix_if_present(
        state_dict,
        prefixes=[
            "module.convnet.",   # 例子
            "module.encoder.",   # 例子
            "encoder.",          # 例子
            "convnet.",          # 例子
            "module.",           # 最后再试通用 module.
        ],
    )

    # 只拷贝 visual_encoder 对应部分
    new_vars = unfreeze(variables)
    p = new_vars["params"]["visual_encoder"]
    bs = new_vars["batch_stats"]["visual_encoder"]

    # stem
    if "conv1.weight" in state_dict:
        p["conv1"]["kernel"] = _torch_conv_to_flax(state_dict["conv1.weight"])
    if "bn1.weight" in state_dict:
        p["bn1"]["scale"] = _torch_vec(state_dict["bn1.weight"])
    if "bn1.bias" in state_dict:
        p["bn1"]["bias"] = _torch_vec(state_dict["bn1.bias"])
    if "bn1.running_mean" in state_dict:
        bs["bn1"]["mean"] = _torch_vec(state_dict["bn1.running_mean"])
    if "bn1.running_var" in state_dict:
        bs["bn1"]["var"] = _torch_vec(state_dict["bn1.running_var"])

    # layer1..layer4，每层两个 BasicBlock
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            flax_block = f"layer{layer_idx}_{block_idx}"
            torch_prefix = f"layer{layer_idx}.{block_idx}"

            # conv1 / bn1
            k = f"{torch_prefix}.conv1.weight"
            if k in state_dict:
                p[flax_block]["conv1"]["kernel"] = _torch_conv_to_flax(state_dict[k])

            k = f"{torch_prefix}.bn1.weight"
            if k in state_dict:
                p[flax_block]["bn1"]["scale"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn1.bias"
            if k in state_dict:
                p[flax_block]["bn1"]["bias"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn1.running_mean"
            if k in state_dict:
                bs[flax_block]["bn1"]["mean"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn1.running_var"
            if k in state_dict:
                bs[flax_block]["bn1"]["var"] = _torch_vec(state_dict[k])

            # conv2 / bn2
            k = f"{torch_prefix}.conv2.weight"
            if k in state_dict:
                p[flax_block]["conv2"]["kernel"] = _torch_conv_to_flax(state_dict[k])

            k = f"{torch_prefix}.bn2.weight"
            if k in state_dict:
                p[flax_block]["bn2"]["scale"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn2.bias"
            if k in state_dict:
                p[flax_block]["bn2"]["bias"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn2.running_mean"
            if k in state_dict:
                bs[flax_block]["bn2"]["mean"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.bn2.running_var"
            if k in state_dict:
                bs[flax_block]["bn2"]["var"] = _torch_vec(state_dict[k])

            # downsample
            k = f"{torch_prefix}.downsample.0.weight"
            if k in state_dict and "downsample_conv" in p[flax_block]:
                p[flax_block]["downsample_conv"]["kernel"] = _torch_conv_to_flax(state_dict[k])

            k = f"{torch_prefix}.downsample.1.weight"
            if k in state_dict and "downsample_bn" in p[flax_block]:
                p[flax_block]["downsample_bn"]["scale"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.downsample.1.bias"
            if k in state_dict and "downsample_bn" in p[flax_block]:
                p[flax_block]["downsample_bn"]["bias"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.downsample.1.running_mean"
            if k in state_dict and "downsample_bn" in bs[flax_block]:
                bs[flax_block]["downsample_bn"]["mean"] = _torch_vec(state_dict[k])
            k = f"{torch_prefix}.downsample.1.running_var"
            if k in state_dict and "downsample_bn" in bs[flax_block]:
                bs[flax_block]["downsample_bn"]["var"] = _torch_vec(state_dict[k])

    return freeze(new_vars)