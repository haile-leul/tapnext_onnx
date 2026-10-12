"""Export-friendly wrappers around Track_On2 (two graphs).

Track-On2 is split exactly where the official streaming Predictor splits it:

  TrackOnEncoder  frame -> multi-scale features (DINOv3 ViT-S+ + ViT-Adapter + FPN).
                  Runs once per frame; cost independent of the number of points.

  TrackOnTracker  features + per-point state -> positions, visibility, next state.
                  N (number of points) is a dynamic dimension.

Per-point state lives OUTSIDE the graph (rows of q_init / point_memory /
memory_mask), exactly as in model/trackon_predictor.py, so the caller can add
points (append rows, mark is_new=1) and remove points (drop rows) at any frame.

Re-implemented here (same submodules / weights, same maths) only where the
original code is not traceable: asserts in indices_to_coords, data-dependent
branches in MHA_Block, nn.MultiheadAttention fast paths.
"""

import torch
from torch import nn
from torch.nn import functional as F

NEG = -30000.0  # masked attention logit (finite so fp16 engines stay NaN-free)


def mha(m, q, k, v, key_mask=None):
  """nn.MultiheadAttention (batch_first, eval) written out explicitly."""
  e, h = m.embed_dim, m.num_heads
  d = e // h
  w, b = m.in_proj_weight, m.in_proj_bias
  qq = F.linear(q, w[:e], b[:e])
  kk = F.linear(k, w[e:2 * e], b[e:2 * e])
  vv = F.linear(v, w[2 * e:], b[2 * e:])
  bsz, lq, _ = qq.shape
  lk = kk.shape[1]
  qq = qq.view(bsz, lq, h, d).transpose(1, 2)
  kk = kk.view(bsz, lk, h, d).transpose(1, 2)
  vv = vv.view(bsz, lk, h, d).transpose(1, 2)
  att = torch.matmul(qq * (d ** -0.5), kk.transpose(-1, -2))  # scale first: fp16-safe
  if key_mask is not None:  # (bsz, lk) bool, True = ignore
    att = att.masked_fill(key_mask[:, None, None, :], NEG)
  out = torch.matmul(att.softmax(-1), vv).transpose(1, 2).reshape(bsz, lq, e)
  return m.out_proj(out)


def l2_normalize(x):
  """F.normalize(x, dim=-1) without fp16 overflow: sum(x^2) over 256 channels
  can exceed 65504, so normalise by max|x| first (identical result for x != 0)."""
  xs = x / x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
  return xs / torch.linalg.vector_norm(xs, dim=-1, keepdim=True).clamp_min(1e-12)


def mha_block(blk, q, k, v, key_mask=None):
  """model.modules.MHA_Block.forward in eval mode (the 'all keys masked' branch
  never triggers here: the current query token is always unmasked)."""
  q = blk.norm1(q + mha(blk.mha, q, k, v, key_mask))
  return blk.norm2(q + blk.linear2(F.relu(blk.linear1(q))))


class TrackOnEncoder(nn.Module):
  """frame [1,3,H,W] RGB float 0..255 at the model input size (384x512) ->
  f4 [1,P,D], f8 [1,P/4,D], f16 [1,P/16,D], f32 [1,P/64,D], f_fused [1,P,D]."""

  def __init__(self, model):
    super().__init__()
    self.m = model

  def forward(self, frame):
    x = (frame / 255.0).unsqueeze(1)
    f4, f8, f16, f32 = self.m.backbone(x)
    ff = self.m.fpn(f4, f8, f16, f32)
    return f4[0], f8[0], f16[0], f32[0], ff[0]


class TrackOnTracker(nn.Module):
  """One tracking step for N points (Track_On2.track_frame + Predictor bookkeeping).

  inputs   f4, f8, f16, f32, f_fused    encoder outputs of the current frame
           q_init      [N, D]   query features (rows of new points: anything)
           memory      [N, M, D] point memory
           memory_mask [N, M]   1 = empty memory slot, 0 = filled
           new_xy      [N, 2]   (x, y) in model-input pixels (512x384) for new points
           is_new      [N]      1 for points added on this frame, else 0
  outputs  points      [N, 2]   (x, y) in model-input pixels
           vis_logit   [N]      visible if sigmoid >= delta_v (0.8)
           unc_logit   [N]      uncertainty logit
           q_init_out, memory_out, memory_mask_out  -> next frame's state
  """

  def __init__(self, model):
    super().__init__()
    self.m = model
    self.h, self.w = model.input_size
    self.hf, self.wf, self.d, self.stride = model.Hf, model.Wf, model.D, model.stride

  def correlation(self, q, h4, h8, h16, h32):
    """Track_On2.multiscale_correlation (cosine similarity at 4 scales -> 1x1 conv)."""
    m, n = self.m, q.shape[1]
    qn = l2_normalize(q)

    def corr(h, hs, ws):
      return torch.matmul(qn, l2_normalize(h).transpose(1, 2)).view(n, 1, hs, ws)

    size = (self.hf, self.wf)
    up = lambda c: F.interpolate(c, size=size, mode='bilinear', align_corners=False)
    c = torch.cat([corr(h4, self.hf, self.wf), up(corr(h8, self.hf // 2, self.wf // 2)),
                   up(corr(h16, self.hf // 4, self.wf // 4)), up(corr(h32, self.hf // 8, self.wf // 8))], 1)
    return m.ms_corr_proj(c).view(1, n, m.P)

  def idx_to_xy(self, idx):
    rows = torch.div(idx, self.wf, rounding_mode='floor')
    cols = idx - rows * self.wf
    return torch.stack([cols.float() * self.stride + 0.5 * self.stride,
                        rows.float() * self.stride + 0.5 * self.stride], -1)

  def sample_queries(self, f_fused, xy):
    fmap = f_fused.view(1, self.hf, self.wf, self.d).permute(0, 3, 1, 2)
    g = torch.stack([xy[:, 0] / self.w, xy[:, 1] / self.h], -1) * 2 - 1
    s = F.grid_sample(fmap, g.view(1, -1, 1, 2), mode='bilinear', padding_mode='border',
                      align_corners=False)
    return s[0, :, :, 0].transpose(0, 1)  # [N, D]

  def rerank(self, q_t, f4, f8, f16, f32, c_t):
    r = self.m.reranking_head
    n, k, d = q_t.shape[1], r.K, r.D
    top = torch.topk(c_t, k, dim=-1)[1]                               # (1, N, K)
    p_top = self.idx_to_xy(top)                                        # (1, N, K, 2)
    scale = torch.tensor([self.w, self.h], dtype=q_t.dtype, device=q_t.device)
    ref = (p_top / scale).clamp(0, 1).view(1, n * k, 1, 2).expand(-1, -1, r.num_level, -1)
    f_scales = torch.cat([f4, f8, f16, f32], dim=1)
    q_top = q_t.unsqueeze(2).expand(-1, -1, k, -1).reshape(1, n * k, d)
    for layer in r.local_decoder:
      q_top = layer(q_top, f_scales, f_scales, ref, r.spatial_shapes, r.start_levels)
    q_top = torch.cat([q_top.view(1, n, k, d), q_t.unsqueeze(2).expand(-1, -1, k, -1)], -1)
    q_top = r.fusion_layer(q_top).view(n, k, d)
    qv = q_t.view(n, 1, d)
    q = mha_block(r.fusion, qv, q_top, q_top)
    return r.final_projection_layer(torch.cat([q, qv], -1)).view(1, n, d)

  def prediction_head(self, q_t, f4, f8, f16, f32, p_patch):
    hd = self.m.prediction_head
    scale = torch.tensor([self.w, self.h], dtype=q_t.dtype, device=q_t.device)
    ref = (p_patch / scale).clamp(0, 1).unsqueeze(2).expand(-1, -1, hd.num_level, -1)
    f_scales = torch.cat([f4, f8, f16, f32], dim=1)
    q_off = q_t
    for layer in hd.offset_transformer:
      q_off = layer(q_off, f_scales, f_scales, ref, hd.spatial_shapes, hd.start_levels)
    offset = hd.offset_layer(q_off)[0] * hd.stride                    # last layer's offset (N, 2)
    v = hd.vis_layer(q_t)[0, :, 0]
    u = hd.unc_layer(q_t)[0, :, 0]
    return offset, v, u

  def forward(self, f4, f8, f16, f32, f_fused, q_init, memory, memory_mask, new_xy, is_new):
    m = self.m
    n, d = q_init.shape
    q_init = torch.where(is_new[:, None] > 0.5, self.sample_queries(f_fused, new_xy), q_init)
    mask = torch.cat([memory_mask > 0.5, torch.zeros_like(memory_mask[:, :1], dtype=torch.bool)], 1)

    q_t = q_init.unsqueeze(0)
    mem = memory
    for i in range(m.decoder_layer_num):
      q_t = mha_block(m.feature_attention[i], q_t, f_fused, f_fused)
      q_t = mha_block(m.query_attention[i], q_t, q_t, q_t)
      qkv = torch.cat([mem, q_t.view(n, 1, d)], dim=1)
      qkv = mha_block(m.memory_attention[i], qkv + m.t_embedding, qkv + m.t_embedding, qkv, mask)
      q_t = qkv[:, -1].unsqueeze(0)
      mem = qkv[:, :-1]
    q_t = m.projection1(q_t)

    c1 = self.correlation(q_t, f4, f8, f16, f32)
    q_t = m.projection2(self.rerank(q_t, f4, f8, f16, f32, c1))
    c2 = self.correlation(q_t, f4, f8, f16, f32)
    p_patch = self.idx_to_xy(torch.argmax(c2, dim=-1))               # (1, N, 2)
    offset, v_logit, u_logit = self.prediction_head(q_t, f4, f8, f16, f32, p_patch)
    points = p_patch[0] + offset

    memory_out = torch.cat([memory[:, 1:], q_t.view(n, 1, d)], dim=1)
    mask_out = torch.cat([memory_mask[:, 1:], torch.zeros_like(memory_mask[:, :1])], dim=1)
    return points, v_logit, u_logit, q_init, memory_out, mask_out
