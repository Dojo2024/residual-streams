import sys
from pathlib import Path

# make tests/paper_reference.py and the package importable without installing
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
