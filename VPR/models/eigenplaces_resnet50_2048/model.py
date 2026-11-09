"""EigenPlaces ResNet50 / 2048 pretrained on SF-XL."""
import torch
from VPR.models.base import VPRExtractor, configure_hub

MODEL_ID = 'eigenplaces_resnet50_2048'
DIM = 2048


def load(device=None):
    configure_hub()
    print(f'[MODEL] {MODEL_ID}: loading pretrained weights', flush=True)
    model = torch.hub.load('gmberton/eigenplaces:main', 'get_trained_model',
                           backbone='ResNet50', fc_output_dim=DIM, trust_repo=True)
    return VPRExtractor(MODEL_ID, DIM, model, device=device)
