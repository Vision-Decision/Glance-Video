"""Synthetic inserted events: a word card over the frame (M7 / M8 memory probes)."""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

WORDS = ("apple banana guitar rocket tiger pizza castle dolphin piano camera violin tractor penguin umbrella volcano "
         "giraffe helmet lemon robot anchor balloon candle diamond eagle forest hammer island jacket kettle ladder "
         "magnet needle orange parrot rabbit saddle tomato wallet zebra bridge").split()
COLORS = [(220, 40, 40), (40, 90, 220), (30, 160, 60), (230, 160, 20), (140, 50, 180)]


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default(size=sz)


def card(frames, word, color, size):
    """overlay a centred colour box with the word in white on every frame (uint8 (T,3,S,S))"""
    f = font(int(size * 0.16))
    out = []
    for x in frames:
        im = Image.fromarray(x.permute(1, 2, 0).numpy()); d = ImageDraw.Draw(im)
        w, h = d.textbbox((0, 0), word.upper(), font=f)[2:]
        pad = int(size * 0.05); x0, y0 = (size - w) // 2 - pad, (size - h) // 2 - pad
        d.rectangle([x0, y0, x0 + w + 2 * pad, y0 + h + 2 * pad], fill=color)
        d.text((x0 + pad, y0 + pad // 2), word.upper(), fill=(255, 255, 255), font=f)
        out.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
    return torch.stack(out)
