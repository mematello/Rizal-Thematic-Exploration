"""Create square favicons from the provided portrait without cropping or distortion.
Requires Pillow: python -m pip install Pillow
"""
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
source = ImageOps.exif_transpose(Image.open(ROOT / 'logo.jpg')).convert('RGBA')

def icon(size):
    scaled = ImageOps.contain(source, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(scaled, ((size-scaled.width)//2, (size-scaled.height)//2))
    return canvas

public = ROOT / 'frontend/public'
for size in (16, 32, 48, 192, 512):
    icon(size).save(public / f'favicon-{size}x{size}.png')
icon(512).save(public / 'icon.png')
icon(180).save(public / 'apple-icon.png')
icon(256).save(public / 'favicon.ico', format='ICO',
               sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)])
