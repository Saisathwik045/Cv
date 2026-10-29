"""Generate synthetic workbook/images at runtime; never represents real ODIR data."""
from pathlib import Path
import random
from openpyxl import Workbook
from PIL import Image
from trustretina.config import load_config


def make_fixture(folder, count=160):
    root = Path(folder) / 'SYNTHETIC_ODIR_FIXTURE'
    root.mkdir()
    images = root / 'Training Images'
    images.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'SyntheticLabels'
    sheet.append(['ID', 'Left-Fundus', 'Right-Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    marker = workbook.create_sheet('TRUSTRETINA_SYNTHETIC_FIXTURE')
    marker.append(['SYNTHETIC TEST DATA ONLY; NOT REAL RETINAL IMAGES OR DISEASE RESULTS'])
    rng = random.Random(300)
    for i in range(count):
        sheet.append([i, f'{i}_left.png', f'{i}_right.png', *[int(i % 8 == j) for j in range(8)]])
        for eye in ('left', 'right'):
            pixels = bytes(rng.getrandbits(8) for _ in range(32 * 32 * 3))
            Image.frombytes('RGB', (32, 32), pixels).save(images / f'{i}_{eye}.png')
    workbook.save(root / 'data.xlsx')
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/phase1.toml')
    config['data']['root'] = str(root)
    config['data']['kind'] = 'synthetic_fixture'
    config['data']['duplicate_policy'] = 'reject'
    return config
