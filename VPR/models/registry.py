"""Single source of truth for available VPR models."""
from importlib import import_module

MODEL_IDS = (
    'eigenplaces_resnet18_256',
    'eigenplaces_resnet50_2048',
    'cosplace_resnet18_512',
    'mixvpr_resnet50_512',
)
MODEL_DIMS = dict(zip(MODEL_IDS, (256, 2048, 512, 512)))


def load_model(name: str, device=None):
    if name not in MODEL_IDS:
        raise ValueError(f'Unknown model: {name}. Valid: {", ".join(MODEL_IDS)}')
    module = import_module(f'VPR.models.{name}.model')
    return module.load(device=device)
