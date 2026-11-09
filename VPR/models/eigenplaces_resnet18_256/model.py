"""EigenPlaces ResNet18 / 256. Bilinear resize matches the initial Node-VPR DB."""
import torch
from VPR.models.base import VPRExtractor, configure_hub

MODEL_ID = 'eigenplaces_resnet18_256'
DIM = 256


def load(device=None):
    configure_hub()
    print(f'[MODEL] {MODEL_ID}: loading pretrained weights', flush=True)
    model = torch.hub.load('gmberton/eigenplaces:main', 'get_trained_model',
                           backbone='ResNet18', fc_output_dim=DIM, trust_repo=True)
    return VPRExtractor(MODEL_ID, DIM, model, device=device)
