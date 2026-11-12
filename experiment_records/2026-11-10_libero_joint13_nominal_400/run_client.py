"""Run the frozen physical evaluation entry with bounded transport recovery only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from transport_reconnect import install
install()
from examples.libero import eval_joint13_nominal_libero_local as entry
# Keep child launches on this same transport wrapper; no evaluation source changes.
entry.__file__=str(Path(__file__).resolve())
entry.main()
