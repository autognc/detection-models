# -*- coding: utf-8 -*-
"""Top-level package for detection-models."""

__author__ = """Gavin C. Martin"""
__email__ = 'gavinmartin@utexas.edu'
__version__ = '0.1.1'

from pathlib import Path
import os
import sys

outer_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(
    str((outer_dir / ".deps" / "models" / "research" / "slim").absolute()))
sys.path.append(str((outer_dir / ".deps" / "models" / "research").absolute()))