from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def make_image(label: int, size: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = Image.new('RGB', (size, size), (20, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, size - 8, size - 8), fill=(115, 45, 38))
    draw.ellipse((74, 50, 88, 64), fill=(230, 190, 120))
    for _ in range(15):
        draw.line((82, 57, int(rng.integers(15, 115)), int(rng.integers(15, 115))), fill=(150, 60, 50), width=1)
    for _ in range(label * 4):
        x = int(rng.integers(20, 108))
        y = int(rng.integers(20, 108))
        r = int(rng.integers(2, 5))
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(220, 35, 35))
    return img


def generate(samples=250, output_dir='data/demo_images'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(samples):
        label = i % 5
        image = make_image(label, seed=42 + i)
        filename = f'image_{i:04d}.png'
        image.save(output / filename)
        rows.append({'image_path': str(output / filename), 'label': label})
    pd.DataFrame(rows).to_csv(output / 'labels.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=250)
    parser.add_argument('--output-dir', default='data/demo_images')
    args = parser.parse_args()
    generate(args.samples, args.output_dir)
    print(args.output_dir)


if __name__ == '__main__':
    main()
