"""Validate every required COLMAP mask, failing rather than falling back silently."""
import hashlib
from pathlib import Path
from PIL import Image


def validate_masks(image_root, names, mask_root):
    image_root, mask_root = Path(image_root), Path(mask_root).resolve()
    records = []
    for name in names:
        path = (mask_root / (name + '.png')).resolve()
        if not path.is_relative_to(mask_root) or not path.is_file():
            raise ValueError(f'Missing or unsafe mask: {name}')
        with Image.open(image_root / name) as original, Image.open(path) as mask:
            if mask.mode != 'L' or mask.size != original.size:
                raise ValueError(f'Mask dimensions/mode mismatch: {name}')
            histogram = mask.histogram()
            if any(histogram[1:255]) or histogram[255] == 0:
                raise ValueError(f'Mask is nonbinary or empty: {name}')
            records.append({'image': name, 'mask': str(path),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'retained_fraction': histogram[255]/sum(histogram)})
    return {'mask_root': str(mask_root), 'count': len(records), 'records': records,
            'meaning': 'Format/integrity checks only; not segmentation accuracy'}
