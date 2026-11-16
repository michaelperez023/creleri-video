import sys
from pathlib import Path

# api.py and segment_utils.py are run as top-level modules from adatad_temp_seg/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
