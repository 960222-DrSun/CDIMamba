"""CDIMamba single-image inference script.

Loads a trained checkpoint and restores one image (or a folder of images).

Usage
-----
    python inference_cdimmaba.py \
        --checkpoint experiments/CDIMamba_GoPro/models/net_g_latest.pth \
        --input  datasets/GoPro/test/blur \
        --output results/deblur

The script has no dependency on ``basicsr``'s training machinery: it builds the
architecture straight from ``models.arch`` and handles crop-tiling so that
arbitrary resolutions fit in memory.
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = osp.dirname(osp.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models  # noqa: E402,F401  registers the architectures
from models.arch import CDIMamba  # noqa: E402

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_network(checkpoint, device, dim=48, num_blocks=(4, 6, 6), num_heads=(1, 2, 4),
                 **arch_kwargs):
    """Build CDIMamba and load ``checkpoint`` (raw state_dict or BasicSR ckpt)."""
    net = CDIMamba(dim=dim, num_blocks=list(num_blocks), num_heads=list(num_heads),
                   **arch_kwargs)
    state = torch.load(checkpoint, map_location='cpu')
    # BasicSR checkpoints wrap the weights under 'params' (or 'params_ema')
    for key in ('params_ema', 'params', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys, first few: {missing[:5]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys, first few: {unexpected[:5]}')
    return net.to(device).eval()


def _infer_single(net, img, device, tile=None, tile_overlap=32):
    """Run the network on an RGB float32 image in [0, 1]."""
    h, w, _ = img.shape
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        if tile is None or (h <= tile and w <= tile):
            out = net(tensor)
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # simple overlapped tiling for large images
        out = np.zeros_like(img)
        weight = np.zeros((h, w, 1), dtype=np.float32)
        stride = tile - tile_overlap
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, x1 = min(y + tile, h), min(x + tile, w)
                y0, x0 = max(y1 - tile, 0), max(x1 - tile, 0)
                patch = tensor[:, :, y0:y1, x0:x1]
                pred = net(patch).squeeze(0).permute(1, 2, 0).cpu().numpy()
                out[y0:y1, x0:x1] += pred
                weight[y0:y1, x0:x1] += 1.0
        return out / np.maximum(weight, 1.0)


def main():
    parser = argparse.ArgumentParser(description='CDIMamba inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True,
                        help='image file or folder of images')
    parser.add_argument('--output', type=str, default='results/deblur')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_directions', type=int, default=4)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--tile', type=int, default=None,
                        help='process large images in tiles of this size')
    parser.add_argument('--tile_overlap', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    net = load_network(args.checkpoint, device, dim=args.dim,
                       num_directions=args.num_directions,
                       num_experts=args.num_experts, topk=args.topk)
    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'CDIMamba loaded: {num_params:.2f} M parameters (paper: 4.41 M)')

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        files = [in_path]
    if not files:
        raise FileNotFoundError(f'no images found under {in_path}')

    for path in files:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'[skip] cannot read {path}')
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred = _infer_single(net, img, device, args.tile, args.tile_overlap)
        pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / path.name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        print(f'[ok] {path.name}')

    print(f'results written to {out_dir}')


if __name__ == '__main__':
    main()
