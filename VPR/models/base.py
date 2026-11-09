"""Shared descriptor inference interface. Model-specific networks live in subfolders."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class VPRExtractor:
    def __init__(self, model_id: str, dim: int, model, device: str | None = None,
                 interpolation=T.InterpolationMode.BILINEAR):
        self.model_id = model_id
        self.dim = dim
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device).eval()
        self.interpolation = str(interpolation)
        self.transform = T.Compose([
            T.Resize((320, 320), interpolation=interpolation),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])

    @staticmethod
    def get_rgb(image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return ImageOps.exif_transpose(im).convert('RGB')
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert('RGB')
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        raise TypeError(f'Unsupported image type: {type(image)!r}')

    def batch(self, images):
        tensors = [self.transform(self.get_rgb(x)) for x in images]
        if not tensors:
            return np.empty((0, self.dim), dtype=np.float32)
        with torch.inference_mode():
            result = self.model(torch.stack(tensors).to(self.device))
            result = result.flatten(1)
            result = torch.nn.functional.normalize(result.float(), p=2, dim=1)
            output = result.cpu().numpy().astype(np.float32)
        if output.shape != (len(images), self.dim) or not np.isfinite(output).all():
            raise RuntimeError(f'{self.model_id}: invalid descriptor shape/values {output.shape}')
        return output

    def descriptor(self, image):
        return self.batch([image])[0]


def configure_hub():
    path = ROOT / 'Models' / 'torchhub'
    path.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(path))
    return path
