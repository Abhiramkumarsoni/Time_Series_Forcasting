"""
conftest.py

Shared pytest fixtures and path setup.
Ensures 'src' is importable from tests without requiring pip install.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported directly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
