"""Compatibility shim — re-exports the canonical `GoldenExtractor`.

This file used to be a stale fork of the golden model with
`SAMPLE_W=14`, `FFT_W=18`, and `Q_FRAC=12`.  Those constants no longer
match the RTL pipeline (which is now `SAMPLE_W=16`, `FFT_W=16`,
`Q_FRAC=10` after the R2FFT / BFP rewrite), and the fork was used only
by `src/ml/Pipeline/process_data_golden.py`, which silently produced
training features at a different scale than what the hardware actually
computes.

This shim makes the two imports resolve to the same module so there is
one source of truth.  Any script that imports from `src/ml/my_test/`
now gets the canonical implementation at `src/ml/golden_model.py`,
including per-frame BFP emulation and the bfpexp compensation that
matches `pipeline_top.sv`'s `mel_compensated` output.

The `.pt` checkpoints, `dscnn.py`, `export.py`, and other utility
scripts in this directory are untouched.
"""
# The canonical module and this shim share a name (`golden_model`), so we
# cannot use a plain `from golden_model import *` — that would re-enter the
# shim.  Load the canonical file directly by path and copy its public names
# into this module's namespace.
import importlib.util as _ilu
from pathlib import Path as _Path

_canonical_path = _Path(__file__).resolve().parent.parent / "golden_model.py"
_spec = _ilu.spec_from_file_location("_canonical_golden_model", str(_canonical_path))
_canonical = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_canonical)

# Re-export every public attribute (skip dunders) so callers see the same
# namespace they would get from the canonical module.
_locals = globals()
for _name in dir(_canonical):
    if not _name.startswith("_"):
        _locals[_name] = getattr(_canonical, _name)
