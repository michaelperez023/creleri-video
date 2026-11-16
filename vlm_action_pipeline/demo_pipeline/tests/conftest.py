import sys
from pathlib import Path

# The services run as scripts from demo_pipeline/, so import them the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
