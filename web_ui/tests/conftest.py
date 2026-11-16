import sys
from pathlib import Path

# app.py is a script next to this folder rather than an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
