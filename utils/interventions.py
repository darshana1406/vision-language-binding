from utils.data import *
from utils.helper import *

import torch
from jaxtyping import Float, Int, Bool
import einops

from transformer_lens.utils import get_act_name

def replace_cache(src_cache, tgt_cache, src_pos, tgt_pos, num_layers, hook_name, padding=0):
    src_pos = Substring(src_pos[0], src_pos[1]+padding)
    tgt_pos = Substring(tgt_pos[0], tgt_pos[1]+padding)

    for layer in range(num_layers):
        key = get_act_name(hook_name, layer)
        src_cache[key][:,src_pos.to_slice(),:] = tgt_cache[key][:,tgt_pos.to_slice(),:]

    return src_cache

def replace_cache_with0(src_cache, pos, num_layers, hook_name):

    for layer in range(num_layers):
        key = get_act_name(hook_name, layer)
        pos = Substring(pos[0],pos[1]+1)
        src_cache[key][:,pos.to_slice(),:] = 0

    return src_cache


def patch_masked_residue(
    target_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    mask: Bool[torch.Tensor, "batch pos"],
    source_cache,
):
    device = target_residual_component.device
    target_residual_component[mask.to(device), :] = (
        source_cache[hook.name].to(device)[mask.to(device), :].to(device)
    )
    return target_residual_component




def rotary_deltas(
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    pos_deltas: Int[torch.Tensor, "batch pos"],
    attn,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # adapted from components.py -> Attention -> apply_rotary
    x_pos = x.size(1)
    x_rot = x[..., : attn.cfg.rotary_dim]  # batch pos head_index d_head
    x_pass = x[..., attn.cfg.rotary_dim :]
    x_flip = attn.rotate_every_two(x_rot)
    abs_pos = torch.abs(pos_deltas)
    coses = attn.rotary_cos[abs_pos]  # batch pos d_head
    sines = attn.rotary_sin[abs_pos] * torch.sign(pos_deltas)[..., None]
    x_rotated = x_rot * einops.rearrange(
        coses, "b p d -> b p 1 d"
    ) + x_flip * einops.rearrange(sines, "b p d -> b p 1 d")
    return torch.cat([x_rotated, x_pass], dim=-1)


def patch_rotary_k(
    target_k: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos_deltas: Int[torch.Tensor, "batch pos"],
    rotate_function,
):
    # consistency tests:
    # y = rotate_function(target_k, pos_deltas)
    # x = rotate_function(y, -pos_deltas)
    # assert torch.allclose(target_k, x, rtol=1e-3, atol=1e-4)
    return rotate_function(target_k, pos_deltas.to(target_k.device))


def update_cache(src_cache, src_pos, update_val, num_layers, hook_name):

    if isinstance(num_layers, tuple):
        st, end = num_layers
    else:
        st, end = 0,num_layers
    for layer in range(st,end):
        key = get_act_name(hook_name, layer)
        src_cache[key][:,src_pos.to_slice(),:] += update_val[layer]

    return src_cache
