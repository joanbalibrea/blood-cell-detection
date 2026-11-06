"""Single-cell inference on CPU (default), with optional Apple MPS."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from training.common import CLASSES, build_model, transforms_for


class CellClassifier:
    def __init__(self, checkpoint=None, device='cpu'):
        checkpoint = checkpoint or Path(__file__).resolve().parent / 'models/best.pt'
        if device not in {'cpu', 'mps'}:
            raise ValueError('Use cpu or mps for local inference')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable. Use device="cpu".')
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = payload['config']
        if self.config['classes'] != CLASSES or self.config['architecture'] != 'efficientnet_b0':
            raise ValueError('Incompatible architecture or class mapping')
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms_for(self.config)

    def predict(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise TypeError('Provide an image path or PIL image')
        batch = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            p = self.model(batch).softmax(1)[0].cpu().tolist()
        index = max(range(8), key=p.__getitem__)
        return {'predicted_class': CLASSES[index], 'confidence': p[index],
                'probabilities': dict(zip(CLASSES, p))}


def predict_cell(image, checkpoint=None, device='cpu'):
    return CellClassifier(checkpoint, device).predict(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict_cell(args.image, args.checkpoint, args.device), indent=2))
