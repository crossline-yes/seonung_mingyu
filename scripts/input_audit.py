"""Inspect selected source images without modifying or excluding them."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import time

from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.ppm', '.tif', '.tiff'}


def list_images(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def audit_images(root, names):
    start = time.monotonic()
    root = Path(root).resolve()
    records = []
    hashes = defaultdict(list)
    sizes = Counter()
    cameras = Counter()
    for name in names:
        path = (root / name).resolve()
        item = {'name': name}
        try:
            if not path.is_relative_to(root):
                raise ValueError('Image is outside the input directory')
            item['bytes'] = path.stat().st_size
            with path.open('rb') as stream:
                item['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            hashes[item['sha256']].append(name)
            with Image.open(path) as image:
                image.verify()
            # Reopen after verify and decode all pixels to detect truncation.
            with Image.open(path) as image:
                item.update(width=image.width, height=image.height, mode=image.mode)
                exif = image.getexif()
                item['camera_make'] = str(exif.get(271, ''))
                item['camera_model'] = str(exif.get(272, ''))
                item['exif_orientation'] = int(exif.get(274, 1))
                image.load()
                if 'A' in image.getbands():
                    alpha_histogram = image.getchannel('A').histogram()
                    item['alpha_values_present'] = [value for value, frequency in enumerate(alpha_histogram) if frequency]
                    item['fully_opaque_fraction'] = alpha_histogram[255] / (image.width * image.height)
            item['status'] = 'readable'
            sizes[f"{item['width']}x{item['height']}"] += 1
            if item['camera_make'] or item['camera_model']:
                cameras[(item['camera_make'] + ' ' + item['camera_model']).strip()] += 1
        except Exception as error:
            item.update(status='unreadable', error=str(error))
        records.append(item)
    duplicates = [group for group in hashes.values() if len(group) > 1]
    unreadable = [item['name'] for item in records if item['status'] != 'readable']
    warnings = []
    if duplicates:
        warnings.append('Byte-identical duplicate files found; no files were excluded')
    if len(sizes) > 1:
        warnings.append('Multiple image dimensions; review shared-camera assumptions')
    if len(cameras) > 1:
        warnings.append('Multiple EXIF camera identities; review shared-camera assumptions')
    if any(item.get('exif_orientation', 1) != 1 for item in records):
        warnings.append('Non-default EXIF orientation found; check image and mask pixel alignment')
    if unreadable:
        warnings.append('Unreadable source images; baseline must stop before feature extraction')
    return {'kind': 'input_integrity_audit', 'input_root': str(root), 'input_count': len(names),
            'readable_count': len(names) - len(unreadable), 'unreadable_images': unreadable,
            'dimensions': dict(sizes), 'camera_identities': dict(cameras),
            'byte_identical_groups': duplicates, 'warnings': warnings, 'images': records,
            'seconds': time.monotonic() - start,
            'limitation': 'Does not measure blur, object identity, view coverage, or reconstruction suitability'}
