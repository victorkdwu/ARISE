"""Shared bootstrap for the arise package.

Every entry point is run from the repository root as
    ARISE_PRESET=<preset> python -m arise.<script> ...
The preset names the target model (TARGET) and its code directory
(TARGET_DIR, relative to the repository root).  On import this module puts
the repository root and TARGET_DIR on sys.path and changes into TARGET_DIR,
so every relative path in a preset is relative to the target's own directory.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from arise_config import cfg, describe  # noqa: E402

TARGET = cfg('TARGET', None)
TARGET_DIR = cfg('TARGET_DIR', None)
if TARGET not in ('horuseye', 'simpleae', 'kitnet') or not TARGET_DIR:
    raise SystemExit('arise: the preset must set TARGET (horuseye|simpleae|kitnet) and TARGET_DIR')
TARGET_DIR = os.path.join(REPO, TARGET_DIR)
sys.path.insert(0, TARGET_DIR)
os.chdir(TARGET_DIR)

__all__ = ['cfg', 'describe', 'REPO', 'TARGET', 'TARGET_DIR']
