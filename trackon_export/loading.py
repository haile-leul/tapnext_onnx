"""Build Track-On2 / Track-On-R from the official repository + checkpoint.

No mmcv needed (see msda.py). The DINOv3 ViT-S+ backbone weights are gated on
HuggingFace: request access to facebook/dinov3-vits16plus-pretrain-lvd1689m and
run `huggingface-cli login` once; they are then downloaded and cached
automatically. `fake_dinov3=True` builds the same architecture with random
weights (only for testing the export pipeline without HF access).
"""

import argparse
import collections
import os
import sys

import torch

from trackon_export.msda import install_mmcv_shim

# config/test.yaml of the official repo (+ inference-time memory size of Predictor)
DEFAULT_ARGS = dict(
    input_size=[384, 512], M=24, D=256, K=16, decoder_layer_num=3,
    predicton_head_layer_num=3, rerank_layer_num=3, vit_backbone='dinov3_s_plus',
    vit_upsample_factor=1.143, M_i=72, delta_v=0.8, grad_checkpoint=False)

# DINOv3 ViT-S+/16 architecture (random weights for --fake-dinov3 tests)
DINOV3_SPLUS = dict(patch_size=16, hidden_size=384, intermediate_size=1536, num_hidden_layers=12,
                    num_attention_heads=6, hidden_act='silu', use_gated_mlp=True,
                    num_register_tokens=4, layerscale_value=1e-5, rope_theta=100.0)

ALLOWED_MISSING = ('backbone.vit_encoder.dinov2', 'backbone.vit_encoder.dinov3')


def add_repo(trackon_root):
  root = os.path.abspath(trackon_root)
  if not os.path.isfile(os.path.join(root, 'model', 'trackon.py')):
    raise FileNotFoundError(f'{root} is not a clone of https://github.com/gorkaydemir/track_on')
  if root not in sys.path:
    sys.path.insert(0, root)
  install_mmcv_shim()


def load_args(config_path=None):
  args = dict(DEFAULT_ARGS)
  if config_path:
    import yaml  # pylint: disable=g-import-not-at-top
    with open(config_path) as f:
      args.update(yaml.safe_load(f))
  args['grad_checkpoint'] = False
  args.setdefault('M_i', DEFAULT_ARGS['M_i'])
  args.setdefault('delta_v', DEFAULT_ARGS['delta_v'])
  return argparse.Namespace(**args)


class _patch_dinov3_loader:
  """Force eager attention (export friendly, same maths) and optionally random weights."""

  def __init__(self, fake):
    self.fake = fake

  def __enter__(self):
    import transformers  # pylint: disable=g-import-not-at-top
    self.orig = transformers.AutoModel.from_pretrained
    fake = self.fake

    def from_pretrained(name, *a, **kw):
      if fake:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel  # pylint: disable=g-import-not-at-top
        torch.manual_seed(0)
        cfg = DINOv3ViTConfig(**DINOV3_SPLUS)
        cfg._attn_implementation = 'eager'  # pylint: disable=protected-access
        return DINOv3ViTModel(cfg)
      kw.setdefault('attn_implementation', 'eager')
      return self.orig(name, *a, **kw)

    transformers.AutoModel.from_pretrained = from_pretrained
    return self

  def __exit__(self, *exc):
    import transformers  # pylint: disable=g-import-not-at-top
    transformers.AutoModel.from_pretrained = self.orig


def _state_dict(path):
  raw = torch.load(path, map_location='cpu', weights_only=False)
  if isinstance(raw, dict):
    for key in ('state_dict', 'model', 'model_state', 'ema_state_dict'):
      if key in raw and isinstance(raw[key], dict):
        raw = raw[key]
        break
  return collections.OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in raw.items())


def build_model(trackon_root, ckpt=None, config=None, fake_dinov3=False, seed=0):
  """Returns (Track_On2 model in eval mode with inference memory size M_i, args)."""
  add_repo(trackon_root)
  from model.trackon import Track_On2  # pylint: disable=g-import-not-at-top
  args = load_args(config)
  torch.manual_seed(seed)
  with _patch_dinov3_loader(fake_dinov3):
    model = Track_On2(args)
  if ckpt:
    res = model.load_state_dict(_state_dict(ckpt), strict=False)
    if res.unexpected_keys:
      raise RuntimeError(f'unexpected keys in checkpoint: {res.unexpected_keys[:10]}')
    bad = [k for k in res.missing_keys if not k.startswith(ALLOWED_MISSING)]
    if bad:
      raise RuntimeError(f'checkpoint misses tracker weights: {bad[:10]}')
  elif not fake_dinov3:
    print('WARNING: no --ckpt given: the tracker head has RANDOM weights')
  if args.M_i != args.M:  # same inference-time memory extension as Predictor
    model.memory_extension(args.M_i)
  model.eval()
  for p in model.parameters():
    p.requires_grad_(False)
  return model, args
