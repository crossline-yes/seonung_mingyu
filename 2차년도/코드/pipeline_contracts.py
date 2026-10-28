"""Small, fail-closed contracts for importing existing reconstruction artifacts."""
import hashlib
import json
import math
from pathlib import Path
import struct


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprints(paths):
    return {str(Path(p).resolve()): sha256(p) for p in sorted(set(map(Path, paths)))}


def verify_fingerprints(records):
    for path, expected in records.items():
        if sha256(path) != expected:
            raise ValueError(f'Artifact changed: {path}')


def read_names(path, root):
    root = Path(root).resolve()
    names = [n.strip() for n in Path(path).read_text(encoding='utf-8-sig').splitlines() if n.strip()]
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError('At least three unique input names required')
    for name in names:
        target = (root / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f'Missing or unsafe image: {name}')
    return names


def pinhole_cameras(path):
    """Explicitly reject distortion: resizing a distorted mask is not undistortion."""
    cameras = {}
    with Path(path).open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            cid, model, width, height = struct.unpack('<iiQQ', stream.read(24))
            if model not in (0, 1):
                raise ValueError('Fusion mask adapter supports PINHOLE/SIMPLE_PINHOLE only')
            params = struct.unpack('<' + 'd' * (3 if model == 0 else 4), stream.read(24 if model == 0 else 32))
            fx, fy, cx, cy = (params[0], params[0], *params[1:]) if model == 0 else params
            if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
                raise ValueError('Invalid camera dimensions or intrinsics')
            cameras[cid] = {'size': (width, height), 'k': (fx, fy, cx, cy)}
        if len(cameras) != count or stream.read(1):
            raise ValueError('Invalid camera file')
    return cameras


def model_images(path):
    images = {}
    with Path(path).open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            iid, *values = struct.unpack('<i7di', stream.read(64))
            chars = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError('Truncated image name')
                if char == b'\0':
                    break
                chars.extend(char)
            name = chars.decode('utf-8')
            points, = struct.unpack('<Q', stream.read(8))
            stream.seek(points * 24, 1)
            images[name] = {'id': iid, 'pose': values[:7], 'camera': values[7]}
        if len(images) != count or stream.tell() != Path(path).stat().st_size:
            raise ValueError('Invalid image model')
    return images


def validate_scaled_models(source, dense, names):
    source, dense = Path(source), Path(dense)
    a, b = pinhole_cameras(source / 'cameras.bin'), pinhole_cameras(dense / 'cameras.bin')
    x, y = model_images(source / 'images.bin'), model_images(dense / 'images.bin')
    if set(x) != set(names) or set(y) != set(names):
        raise ValueError('v1 requires all selected images in both imported models')
    sizes = {}
    for name in names:
        if x[name]['id'] != y[name]['id'] or not all(math.isclose(u, v, rel_tol=1e-9, abs_tol=1e-9) for u, v in zip(x[name]['pose'], y[name]['pose'])):
            raise ValueError(f'Dense pose differs from selected sparse model: {name}')
        ac, bc = a[x[name]['camera']], b[y[name]['camera']]
        w, h = ac['size']; dw, dh = bc['size']
        expected = [k * scale for k, scale in zip(ac['k'], (dw/w, dh/h, dw/w, dh/h))]
        if not all(math.isclose(u, v, rel_tol=1e-6, abs_tol=1e-6) for u, v in zip(expected, bc['k'])):
            raise ValueError(f'Dense intrinsics are not pure image scaling: {name}')
        sizes[name] = {'source': ac['size'], 'dense': bc['size']}
    return sizes


def ply_counts(path):
    """Header diagnostics only; never a geometric quality claim."""
    counts = {}
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Invalid PLY signature')
        for _ in range(256):
            line = stream.readline()
            if not line:
                break
            if line.strip() == b'end_header':
                if not stream.read(1):
                    raise ValueError('Empty PLY body')
                return counts
            fields = line.decode('ascii').split()
            if len(fields) == 3 and fields[0] == 'element':
                counts[fields[1]] = int(fields[2])
    raise ValueError('Missing PLY header terminator')
