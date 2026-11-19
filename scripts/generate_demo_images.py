#!/usr/bin/env python3
"""Create labelled geometric fixtures locally, never patient images or predictions."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_inference_bundle import private_directory, private_path, digest, RecoveryError


def generate(destination):
    from PIL import Image, ImageDraw
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    private_directory(destination)
    for name, color in [('left', (100, 35, 20)), ('right', (65, 35, 100)),
                        ('replacement', (20, 90, 70))]:
        image = Image.new('RGB', (320, 240), 'black')
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 10, 285, 230), fill=color)
        draw.line((50, 190, 260, 40), fill=(220, 140, 50), width=6)
        draw.text((20, 110), 'SYNTHETIC NON-PATIENT', fill='white')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        data = buffer.getvalue()
        path = destination / (name + '.png')
        if os.path.lexists(path):
            private_path(path)
            if digest(path) != hashlib.sha256(data).hexdigest():
                raise RecoveryError('Existing demonstration image differs; preserved without overwrite')
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        print('Synthetic non-patient fixtures ready: ' + str(generate(args.destination)))
    except Exception:
        sys.exit('Synthetic fixture preparation failed; existing files are preserved.')
