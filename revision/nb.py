# Load the notebook's own library/data cells so the revision reuses the exact original code.
import json, io, contextlib
from pathlib import Path
_NB = Path(__file__).resolve().parent.parent / "phishing_uda.ipynb"
_cells = [''.join(c['source']) for c in json.load(open(_NB))['cells'] if c['cell_type'] == 'code']
def load(keys=("import re, json", "def numeric_features", "def ingest_generated", "def load_all_real", "def build_domains")):
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for k in keys:
            exec(next(s for s in _cells if k in s), g)
    return g
