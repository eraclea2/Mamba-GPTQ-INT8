import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qlinear_cuda import QuantLinear
from utils import find_layers 
from tqdm import tqdm

CPU='cpu'


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)
def make_quant(
    module,
    names,
    bits,
    group_size,
    name="",
    use_triton: bool = False,
    use_marlin: bool = False,
    disable_exllama: bool = True,
    disable_exllamav2: bool = True,
    use_qigen: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
    trainable: bool = False,
    use_tritonv2: bool = False,
):
    # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
    if disable_exllama is None:
        if disable_exllamav2:
            disable_exllama = False
        else:
            disable_exllama = True

    # QuantLinear = dynamically_import_QuantLinear(
    #     use_triton=use_triton,
    #     desc_act=desc_act,
    #     group_size=group_size,
    #     bits=bits,
    #     use_marlin=use_marlin,
    #     disable_exllama=disable_exllama,
    #     disable_exllamav2=disable_exllamav2,
    #     use_qigen=use_qigen,
    #     use_tritonv2=use_tritonv2,
    # )

    if isinstance(module, QuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            # elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
            #     in_features = submodule.weight.shape[0]
            #     out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            if (
                (not (desc_act) or group_size == -1)
                and not use_triton
                and not use_qigen
                and not use_tritonv2
            ):
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    # use_cuda_fp16=use_cuda_fp16,
                    trainable=trainable,
                    weight_dtype=submodule.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    trainable=trainable,
                    weight_dtype=submodule.weight.dtype,
                )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))

def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    warmup_triton: bool = False,
    force_layer_back_to_cpu: bool = False,
    use_marlin: bool = False,
    use_tritonv2: bool = False,
):
    # QuantLinear = dynamically_import_QuantLinear(
    #     use_triton=use_triton,
    #     desc_act=desc_act,
    #     group_size=group_size,
    #     bits=bits,
    #     disable_exllama=False,
    #     disable_exllamav2=True,
    #     use_marlin=use_marlin,
    #     use_tritonv2=use_tritonv2,
    # )

    if force_layer_back_to_cpu:
        model.to(CPU)
    print("Packing model...")
    # logger.info("Packing model...")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
        use_triton=use_triton,
        use_cuda_fp16=use_cuda_fp16,
        desc_act=desc_act,
        disable_exllama=False,
        disable_exllamav2=True,
        use_marlin=use_marlin,
    )
    qlayers = find_layers(model, [QuantLinear])

    pbar = tqdm(qlayers.keys(), leave=True)
    for name in pbar:
        pbar.set_description(f"Packing {name}...", refresh=True)

        quantizers[name], scale, zero, g_idx = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx = (
            layers[name].to(CPU),
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU),
        )
        if QuantLinear.QUANT_TYPE == "marlin":
            qlayers[name].pack(layers[name], scale)
        else:
            qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)
    print("Model packed.")
    # logger.info("Model packed.")
