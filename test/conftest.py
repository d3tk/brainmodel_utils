import sys
from pathlib import Path

# Make the fork importable even when the editable-install .pth is ignored
# (CPython skips .pth files that macOS has flagged as hidden).
REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
