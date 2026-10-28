"""Pinned toolchain for reproducible WP2 experiments."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COLMAP_VERSION = '4.2.0'
COLMAP_ROOT = PROJECT / 'tools' / f'colmap-{COLMAP_VERSION}'
COLMAP_EXE = COLMAP_ROOT / 'bin' / 'colmap.exe'


def colmap_environment():
    env = dict(os.environ)
    env['PATH'] = str(COLMAP_EXE.parent) + os.pathsep + env.get('PATH', '')
    env['QT_PLUGIN_PATH'] = str(COLMAP_ROOT / 'plugins')
    return env
