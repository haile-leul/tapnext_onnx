"""Export BootsTAPNext (256x256, TRecViT-B/8) to a single-frame streaming ONNX graph.

TAPNext is an *online* recurrent tracker: every frame it updates a recurrent
(RG-LRU + causal conv) state for every image patch token and every query point
token. The exported graph therefore runs exactly ONE frame and carries the
recurrent state explicitly as graph inputs / outputs, so a C++ caller just
loops:

    state = zeros
    for each frame:
        outputs, state = step(frame, queries, state, ...)

Graph inputs
  frame          float32 [1, 256, 256, 3]   RGB, HWC, values in [-1, 1]
  query_points   float32 [1, Q, 3]          (t, x, y): t = absolute frame index
                                            at which the point is defined,
                                            x/y = pixel coords in the 256x256
                                            frame. t > step => slot not active
                                            yet ("unknown" token).
  step           float32 [1]                index of the current frame (0,1,2..)
  video_reset    float32 [1]                1.0 on the very first frame (or to
                                            restart the whole tracker), else 0.
  point_reset    float32 [1, Q]             1.0 to wipe the recurrent state of
                                            a point slot (slot recycling), else 0.
  lru_state      float32 [12, 1024+Q, 768]  recurrent state (zeros at start)
  conv_state     float32 [12, 1024+Q, 3, 768]

Graph outputs
  tracks_xy      float32 [1, Q, 2]  predicted point position (x, y), 256-px coords
  visible_prob   float32 [1, Q]     sigmoid(visibility logit)
  certainty      float32 [1, Q]     prob. mass within 8px of the prediction
  lru_state_out  / conv_state_out   feed back into the next call

A point is usually treated as reliably visible when
visible_prob * certainty > 0.5 (the official "use_certainty" eval mode).
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NUM_LAYERS = 12
WIDTH = 768
IMG = 256
PATCH = 8
NUM_PATCHES = (IMG // PATCH) ** 2  # 1024
CONV_W = 4


def _block_diag_linear(mod, x):
  # x: [N, C] -> [N, H, C/H] @ [H, C/H, C/H]
  n = x.shape[0]
  h, bw, _ = mod.w.shape
  x = x.reshape(n, h, bw)
  y = torch.einsum('nhi,hij->nhj', x, mod.w) + mod.b
  return y.reshape(n, h * bw)


def _rms_norm(mod, x):
  var = torch.mean(x * x, dim=-1, keepdim=True)
  return x * torch.rsqrt(var + mod.eps) * (mod.scale + 1)


def _gelu(x):
  return F.gelu(x, approximate='tanh')


def _sincos_at(xi, yi, c):
  """Row of posemb_sincos_2d(IMG, IMG, c) at integer pixel (yi, xi). [Q,1] -> [Q,C]."""
  d = c // 4
  omega = torch.arange(d, dtype=torch.float32) / (d - 1)
  omega = 1.0 / (10_000.0**omega)
  ax = xi * omega.view(1, d)
  ay = yi * omega.view(1, d)
  return torch.cat([torch.sin(ax), torch.cos(ax), torch.sin(ay), torch.cos(ay)], -1)


def _query_posemb(q_xy, c):
  """Analytic replacement for grid_sample(query_pos_embed, align_corners=False).

  Bilinear interpolation of the sin-cos table with zero padding, without
  storing the 256x256xC table (~200 MB) in the graph.
  """
  px = q_xy[:, 0:1] - 0.5
  py = q_xy[:, 1:2] - 0.5
  x0 = torch.floor(px)
  y0 = torch.floor(py)
  wx1 = px - x0
  wy1 = py - y0
  out = 0.0
  for xi, wx in ((x0, 1 - wx1), (x0 + 1, wx1)):
    for yi, wy in ((y0, 1 - wy1), (y0 + 1, wy1)):
      valid = ((xi >= 0) & (xi <= IMG - 1) & (yi >= 0) & (yi <= IMG - 1)).float()
      out = out + _sincos_at(xi, yi, c) * (wx * wy * valid)
  return out


class TAPNextStep(nn.Module):
  """One-frame TAPNext step with explicit recurrent state (export friendly)."""

  def __init__(self, model):
    super().__init__()
    self.m = model

  def _ssm_block(self, blk, x, lru_h, conv_c, reset):
    """Griffin residual block for a single timestep.

    x: [N, C], lru_h: [N, C], conv_c: [N, 3, C], reset: [N, 1] (1 = fresh token)
    """
    keep = 1.0 - reset
    lru_h = lru_h * keep
    conv_c = conv_c * keep.unsqueeze(-1)

    raw_x = x
    xn = _rms_norm(blk.temporal_pre_norm, raw_x)
    rb = blk.recurrent_block
    y = _gelu(rb.linear_y(xn))
    xb = rb.linear_x(xn)

    # causal depthwise conv over time (window 4)
    conv_in = torch.cat([conv_c, xb.unsqueeze(1)], dim=1)  # [N, 4, C]
    xb = (conv_in * rb.conv_1d.w.unsqueeze(0)).sum(1) + rb.conv_1d.b
    new_conv = conv_in[:, 1:]

    # RG-LRU
    lru = rb.rg_lru
    gate_x = torch.sigmoid(_block_diag_linear(lru.input_gate, xb))
    gate_a = torch.sigmoid(_block_diag_linear(lru.a_gate, xb))
    log_a = -8.0 * gate_a * F.softplus(lru.a_param)
    a = torch.exp(log_a)
    a_square = torch.exp(2 * log_a)
    multiplier = torch.sqrt(1 - a_square)
    multiplier = reset + keep * multiplier
    normalized_x = xb * gate_x * multiplier
    new_h = a * lru_h + normalized_x

    xo = rb.linear_out(new_h * y)
    residual = xo + raw_x
    xm = _rms_norm(blk.channel_pre_norm, residual)
    mlp = blk.mlp_block
    up = torch.einsum('nd,cdD->cnD', xm, mlp.ffw_up.w) + mlp.ffw_up.b[:, 0]
    xm = mlp.ffw_down(_gelu(up[0]) * up[1])
    return xm + residual, new_h, new_conv

  def _embed_queries(self, query_points, step):
    m = self.m
    c = m.width
    q_t = query_points[0, :, 0:1] - step  # [Q, 1], relative to current frame
    q_xy = query_points[0, :, 1:3]  # [Q, 2] (x, y)
    pe = _query_posemb(q_xy, c)  # [Q, C]
    point_query = m.point_query_token.view(1, c) + pe
    mask_tok = m.mask_token.view(1, c)
    unknown_tok = m.unknown_token.view(1, c)
    tok = torch.where(q_t == 0, point_query, mask_tok)
    tok = torch.where(q_t > 0, unknown_tok, tok)
    return tok  # [Q, C]

  def _certainty(self, tracks_yx, track_logits, radius=8.0):
    # Equivalent of tapnext_torch_utils.tracker_certainty, written over the full
    # 256 grid instead of a gathered 17x17 window.
    logits_y, logits_x = track_logits.chunk(2, dim=-1)  # [Q, 256]
    sy = F.softmax(logits_y, dim=-1)
    sx = F.softmax(logits_x, dim=-1)
    coords = torch.arange(IMG, dtype=torch.float32) + 0.5
    dy2 = (coords.view(1, IMG) - tracks_yx[:, 0:1]) ** 2  # [Q, 256]
    dx2 = (coords.view(1, IMG) - tracks_yx[:, 1:2]) ** 2
    inside = (dy2.unsqueeze(2) + dx2.unsqueeze(1)) <= (radius**2 + 1e-8)
    mass = sy.unsqueeze(2) * sx.unsqueeze(1) * inside.float()
    return mass.sum(dim=(1, 2))

  def forward(self, frame, query_points, step, video_reset, point_reset,
              lru_state, conv_state):
    m = self.m
    q = query_points.shape[1]
    img = frame.permute(0, 3, 1, 2)  # [1, 3, H, W]
    vt = m.lin_proj(img)  # [1, C, 32, 32]
    vt = vt.flatten(2).transpose(1, 2)[0]  # [1024, C]
    vt = vt + m.image_pos_emb[0]
    pt = self._embed_queries(query_points, step)
    x = torch.cat([vt, pt], dim=0)  # [N, C]

    reset = torch.cat([
        video_reset.view(1, 1).expand(NUM_PATCHES, 1),
        torch.maximum(point_reset.view(q, 1), video_reset.view(1, 1)),
    ], dim=0)  # [N, 1]

    new_lru, new_conv = [], []
    for i, blk in enumerate(m.blocks):
      x, h, cc = self._ssm_block(blk.ssm_block, x, lru_state[i],
                                 conv_state[i], reset)
      new_lru.append(h)
      new_conv.append(cc)
      x = blk.vit_block(x.unsqueeze(0))[0]
    x = m.encoder_norm(x)
    pts = x[NUM_PATCHES:]

    track_logits = m.coordinate_head(pts)  # [Q, 512]  (y logits, x logits)
    soft_argmax_threshold = 20
    pos_y, pos_x = track_logits.chunk(2, dim=-1)
    index = torch.arange(IMG, dtype=torch.float32).view(1, IMG)

    def soft_argmax(logits):
      am = logits.argmax(dim=-1, keepdim=True).float()
      mask = (torch.abs(am - index) <= soft_argmax_threshold).float()
      p = F.softmax(logits * 0.5, dim=-1) * mask
      p = p / p.sum(dim=-1, keepdim=True)
      return (p * index).sum(-1, keepdim=True) + 0.5

    ty = soft_argmax(pos_y)
    tx = soft_argmax(pos_x)
    tracks_yx = torch.cat([ty, tx], dim=-1)
    vis_logit = m.visible_head(pts)[:, 0]
    cert = self._certainty(tracks_yx, track_logits)
    return (
        torch.cat([tx, ty], dim=-1).unsqueeze(0),
        torch.sigmoid(vis_logit).unsqueeze(0),
        cert.unsqueeze(0),
        torch.stack(new_lru, 0),
        torch.stack(new_conv, 0),
    )


def load_model(tapnet_root, ckpt):
  sys.path.insert(0, tapnet_root)
  from tapnet.tapnext.tapnext_torch import TAPNext  # pylint: disable=g-import-not-at-top
  from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint  # pylint: disable=g-import-not-at-top
  model = TAPNext(image_size=(IMG, IMG))
  model = restore_model_from_jax_checkpoint(model, ckpt)
  model.eval()
  for p in model.parameters():
    p.requires_grad = False
  return model


def dummy_inputs(q):
  n = NUM_PATCHES + q
  return (
      torch.zeros(1, IMG, IMG, 3),
      torch.zeros(1, q, 3),
      torch.zeros(1),
      torch.ones(1),
      torch.zeros(1, q),
      torch.zeros(NUM_LAYERS, n, WIDTH),
      torch.zeros(NUM_LAYERS, n, CONV_W - 1, WIDTH),
  )


INPUT_NAMES = ['frame', 'query_points', 'step', 'video_reset', 'point_reset',
               'lru_state', 'conv_state']
OUTPUT_NAMES = ['tracks_xy', 'visible_prob', 'certainty', 'lru_state_out',
                'conv_state_out']


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tapnet_root', required=True)
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', default='tapnext_step_256.onnx')
  ap.add_argument('--opset', type=int, default=18)
  args = ap.parse_args()

  model = load_model(args.tapnet_root, args.ckpt)
  # Replaced by the analytic _query_posemb; drop it so it is not stored.
  del model._buffers['query_pos_embed']
  step_model = TAPNextStep(model).eval()
  q = torch.export.Dim('num_queries', min=1, max=4096)
  dynamic_shapes = {
      'frame': None,
      'query_points': {1: q},
      'step': None,
      'video_reset': None,
      'point_reset': {1: q},
      'lru_state': {1: q + NUM_PATCHES},
      'conv_state': {1: q + NUM_PATCHES},
  }
  with torch.no_grad():
    onnx_prog = torch.onnx.export(
        step_model,
        dummy_inputs(64),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_shapes=dynamic_shapes,
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        optimize=True,
    )
  onnx_prog.save(args.out, external_data=False)
  print('saved', args.out, os.path.getsize(args.out) / 1e6, 'MB')


if __name__ == '__main__':
  main()
