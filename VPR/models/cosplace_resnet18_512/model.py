"""CosPlace ResNet18 / 512 pretrained on SF-XL."""
import torch
from VPR.models.base import VPRExtractor, configure_hub

MODEL_ID = 'cosplace_resnet18_512'
DIM = 512


def load(device=None):
    configure_hub()
    print(f'[MODEL] {MODEL_ID}: loading pretrained weights', flush=True)
    model = torch.hub.load('gmberton/cosplace:main', 'get_trained_model',
                           backbone='ResNet18', fc_output_dim=DIM, trust_repo=True)
    return VPRExtractor(MODEL_ID, DIM, model, device=device)
