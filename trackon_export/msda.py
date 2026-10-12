"""Pure-PyTorch drop-in for mmcv.ops.MultiScaleDeformableAttention.

Track-On2 uses mmcv's deformable attention (a custom CUDA op) in the ViT-Adapter,
the re-ranking module and the prediction head. That op cannot be exported to
ONNX / TensorRT, and mmcv is painful to install (especially on Windows).

This module has exactly the same parameters (state_dict keys), the same
forward semantics (including mmcv's `+ identity` residual) and computes the
same function with F.grid_sample (bilinear, zero padding, align_corners=False),
which exports to ONNX GridSample and TensorRT IGridSampleLayer.

`install_mmcv_shim()` registers it as `mmcv.ops.MultiScaleDeformableAttention`
so the unmodified Track-On repository imports and runs without mmcv.
"""

import math
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


def ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
  """value: (bs, num_value, heads, ch); spatial_shapes: list of (H, W) python ints;
  sampling_locations: (bs, nq, heads, levels, points, 2) in [0, 1];
  attention_weights: (bs, nq, heads, levels, points). Returns (bs, nq, heads*ch)."""
  bs, _, num_heads, ch = value.shape
  _, nq, _, num_levels, num_points, _ = sampling_locations.shape
  value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
  grids = 2 * sampling_locations - 1
  sampled = []
  for lvl, (h, w) in enumerate(spatial_shapes):
    v = value_list[lvl].flatten(2).transpose(1, 2).reshape(bs * num_heads, ch, h, w)
    g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # (bs*heads, nq, points, 2)
    sampled.append(F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False))
  attn = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, nq, num_levels * num_points)
  out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1)
  return out.view(bs, num_heads * ch, nq).transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
  """Same constructor, parameters and forward contract as mmcv 2.x."""

  def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
               im2col_step=64, dropout=0.1, batch_first=False, norm_cfg=None,
               init_cfg=None, value_proj_ratio=1.0):
    super().__init__()
    del im2col_step, norm_cfg, init_cfg
    if embed_dims % num_heads != 0:
      raise ValueError('embed_dims must be divisible by num_heads')
    self.embed_dims, self.num_heads = embed_dims, num_heads
    self.num_levels, self.num_points = num_levels, num_points
    self.batch_first = batch_first
    self.dropout = nn.Dropout(dropout)
    self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
    self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
    value_proj_size = int(embed_dims * value_proj_ratio)
    self.value_proj = nn.Linear(embed_dims, value_proj_size)
    self.output_proj = nn.Linear(value_proj_size, embed_dims)
    # Spatial shapes as python ints, recorded on the first eager call so the
    # module can be traced/exported with static shapes (see freeze_shapes()).
    self.static_shapes = None
    self.init_weights()

  def init_weights(self):
    nn.init.constant_(self.sampling_offsets.weight, 0.)
    thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
    grid = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
    grid = grid.repeat(1, self.num_levels, self.num_points, 1)
    for i in range(self.num_points):
      grid[:, :, i, :] *= i + 1
    with torch.no_grad():
      self.sampling_offsets.bias.copy_(grid.view(-1))
    nn.init.constant_(self.attention_weights.weight, 0.)
    nn.init.constant_(self.attention_weights.bias, 0.)
    nn.init.xavier_uniform_(self.value_proj.weight)
    nn.init.constant_(self.value_proj.bias, 0.)
    nn.init.xavier_uniform_(self.output_proj.weight)
    nn.init.constant_(self.output_proj.bias, 0.)

  def forward(self, query, key=None, value=None, identity=None, query_pos=None,
              key_padding_mask=None, reference_points=None, spatial_shapes=None,
              level_start_index=None, **kwargs):
    del key, level_start_index, kwargs
    if value is None:
      value = query
    if identity is None:
      identity = query
    if query_pos is not None:
      query = query + query_pos
    if not self.batch_first:
      query, value = query.permute(1, 0, 2), value.permute(1, 0, 2)
    if self.static_shapes is not None:
      shapes = self.static_shapes
    else:
      shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    bs, nq, _ = query.shape
    nv = value.shape[1]
    value = self.value_proj(value)
    if key_padding_mask is not None:
      value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.view(bs, nv, self.num_heads, -1)
    offsets = self.sampling_offsets(query).view(
        bs, nq, self.num_heads, self.num_levels, self.num_points, 2)
    attn = self.attention_weights(query).view(bs, nq, self.num_heads, self.num_levels * self.num_points)
    attn = attn.softmax(-1).view(bs, nq, self.num_heads, self.num_levels, self.num_points)
    if reference_points.shape[-1] != 2:
      raise NotImplementedError('only 2-d reference points are used by Track-On2')
    normalizer = torch.tensor([[w, h] for h, w in shapes], dtype=query.dtype, device=query.device)
    loc = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
    out = self.output_proj(ms_deform_attn(value, shapes, loc, attn))
    if not self.batch_first:
      out = out.permute(1, 0, 2)
    return self.dropout(out) + identity


def install_mmcv_shim():
  """Makes `from mmcv.ops import MultiScaleDeformableAttention` resolve to this module."""
  mmcv = sys.modules.get('mmcv') or types.ModuleType('mmcv')
  ops = types.ModuleType('mmcv.ops')
  ops.MultiScaleDeformableAttention = MultiScaleDeformableAttention
  mmcv.ops = ops
  sys.modules['mmcv'] = mmcv
  sys.modules['mmcv.ops'] = ops


def freeze_shapes(model):
  """Record spatial shapes from one eager forward before export: call after a
  forward pass; every MSDA module then uses its python-int shapes."""
  for m in model.modules():
    if isinstance(m, MultiScaleDeformableAttention) and m.static_shapes is None:
      raise RuntimeError('run one eager forward with record_shapes(model) first')


def record_shapes(model):
  """Context: hooks that capture each MSDA module's spatial_shapes as python ints."""
  handles = []

  def hook(mod, args, kwargs):
    ss = kwargs.get('spatial_shapes')
    if ss is not None and mod.static_shapes is None:
      mod.static_shapes = [(int(h), int(w)) for h, w in ss.tolist()]

  for m in model.modules():
    if isinstance(m, MultiScaleDeformableAttention):
      handles.append(m.register_forward_pre_hook(hook, with_kwargs=True))

  class _Ctx:
    def __enter__(self):
      return self

    def __exit__(self, *exc):
      for h in handles:
        h.remove()
  return _Ctx()
