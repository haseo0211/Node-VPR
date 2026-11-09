"""ResNet50 + MixVPR, 512-d, inference-only implementation.

Network dimensions, resize/interpolation, and checkpoint match the official
MixVPR repository (WACV 2023). No Lightning/training dependency required.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models, transforms as T
from VPR.models.base import VPRExtractor, ROOT

MODEL_ID = 'mixvpr_resnet50_512'
DIM = 512
DRIVE_ID = '1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj'
CKPT = ROOT / 'Models' / 'benchmark' / 'resnet50_MixVPR_512_channels(256)_rows(2).ckpt'


class FeatureMixerLayer(nn.Module):
    def __init__(self, width=400):
        super().__init__()
        self.mix = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                 nn.ReLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.mix(x)


class MixAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Sequential(*[FeatureMixerLayer() for _ in range(4)])
        self.channel_proj = nn.Linear(1024, 256)
        self.row_proj = nn.Linear(400, 2)

    def forward(self, x):
        x = self.mix(x.flatten(2))
        x = self.channel_proj(x.permute(0, 2, 1))
        x = self.row_proj(x.permute(0, 2, 1))
        return F.normalize(x.flatten(1), p=2, dim=-1)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv_models.resnet50(weights=None)
        # Retain original module names and state_dict keys.
        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None

    def forward(self, x):
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer3(m.layer2(m.layer1(x)))


class MixVPR512(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.aggregator = MixAggregator()

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def load(device=None):
    if not CKPT.is_file():
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError('Install gdown: .\\.venv\\Scripts\\python.exe -m pip install gdown') from exc
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        print(f'[MODEL] Downloading MixVPR official 512-d checkpoint to {CKPT}', flush=True)
        success = gdown.download(id=DRIVE_ID, output=str(CKPT), quiet=False)
        if not success or not CKPT.is_file() or CKPT.stat().st_size < 1_000_000:
            CKPT.unlink(missing_ok=True)
            raise RuntimeError('MixVPR download failed. Obtain the 512-d checkpoint from official GitHub.')
    print(f'[MODEL] Loading {MODEL_ID} from {CKPT}', flush=True)
    state = torch.load(CKPT, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise ValueError('Unexpected MixVPR checkpoint format')
    for prefix in ('module.', 'model.'):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    model = MixVPR512()
    model.load_state_dict(state, strict=True)
    return VPRExtractor(MODEL_ID, DIM, model, device=device,
                        interpolation=T.InterpolationMode.BICUBIC)
