#!/usr/bin/env python3
"""
update_rtl.py — Apply export.py outputs to RTL files.

Updates:
  1. Shift values (field 11) in LAYER_CFGS in test_kws_top.py  ← from scales.txt
  2. Case statement in bias_DFFs.sv                             ← from bias.hex

Usage (run from src/ml/Pipeline/ or anywhere):
    python3 update_rtl.py [--ckpt-dir <dir>] [--dry-run]

--ckpt-dir : directory containing scales.txt and bias.hex
             (default: same directory as this script)
--dry-run  : print what would change without writing files
"""

import argparse
import re
import sys
from pathlib import Path

HERE      = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "models" / "dscnn-pow2-v12"
REPO_ROOT = HERE.parents[2]   # Pipeline → ml → src → repo root

TEST_FILE = REPO_ROOT / "src/rtl/dscnn/kws_top/test_kws_top.py"
BIAS_SV   = REPO_ROOT / "src/rtl/dscnn/bias_dff/bias_DFFs.sv"

# Architecture-fixed layer order, bias offsets, and channel counts.
# These match the DSCNN(32 filters, 7 classes) architecture and never change
# between training runs — only the values in bias.hex change.
BIAS_LAYERS = [
    ("first_conv",             0,  32),
    ("ds_blocks.0.depthwise", 32,  32),
    ("ds_blocks.0.pointwise", 64,  32),
    ("ds_blocks.1.depthwise", 96,  32),
    ("ds_blocks.1.pointwise",128,  32),
    ("ds_blocks.2.depthwise",160,  32),
    ("ds_blocks.2.pointwise",192,  32),
    ("ds_blocks.3.depthwise",224,  32),
    ("ds_blocks.3.pointwise",256,  32),
    ("classifier",            288,   7),
]
TOTAL_BIASES = sum(n for _, _, n in BIAS_LAYERS)   # 295


# ── helpers ───────────────────────────────────────────────────────────────────

def load_shifts(scales_path: Path) -> list:
    """Return list of int shift values, one per layer, from scales.txt."""
    shifts = []
    with open(scales_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("layer") or s.startswith("-"):
                continue
            shifts.append(int(s.split()[-1]))
    return shifts


def load_biases(bias_path: Path) -> list:
    """Return list of signed int32 values from bias.hex (one 8-digit hex per line)."""
    values = []
    with open(bias_path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = int(s, 16)
            if v >= 0x80000000:
                v -= 0x100000000
            values.append(v)
    return values


# ── update LAYER_CFGS shifts ──────────────────────────────────────────────────

# Matches one LAYER_CFGS tuple line and isolates field 11 (the shift).
# Group 1: open-paren + fields 0-10 (each with trailing comma+spaces)
# Group 2: field 11 value (shift)
# Group 3: rest of tuple (fields 12-15 + closing ),)
_TUPLE_RE = re.compile(
    r'(\(\s*(?:-?\d+\s*,\s*){11})'   # fields 0-10
    r'(-?\d+)'                         # field 11 = shift
    r'(\s*(?:,\s*-?\d+){4}\s*\),)'   # fields 12-15 + close
)


def update_layer_cfgs(test_path: Path, shifts: list, dry_run: bool) -> bool:
    text = test_path.read_text()

    # Isolate the LAYER_CFGS block so we don't accidentally match other tuples.
    block_re = re.compile(r'(LAYER_CFGS\s*=\s*\[)(.*?)(\])', re.DOTALL)
    bm = block_re.search(text)
    if not bm:
        print("ERROR: LAYER_CFGS block not found in test_kws_top.py")
        return False

    block = bm.group(2)
    tuples_found = len(_TUPLE_RE.findall(block))
    if tuples_found != len(shifts):
        print(f"ERROR: found {tuples_found} LAYER_CFGS tuples but {len(shifts)} shifts in scales.txt")
        return False

    # Replace each tuple's shift in order using a stateful closure.
    shift_iter = iter(enumerate(shifts))
    changed = 0

    def replacer(m):
        nonlocal changed
        idx, new_shift = next(shift_iter)
        old_shift = int(m.group(2))
        if old_shift != new_shift:
            layer_name = BIAS_LAYERS[idx][0] if idx < len(BIAS_LAYERS) else f"L{idx}"
            print(f"  layer {idx:2d} ({layer_name}): shift {old_shift} → {new_shift}")
            changed += 1
        return m.group(1) + str(new_shift) + m.group(3)

    new_block = _TUPLE_RE.sub(replacer, block)
    new_text  = text[:bm.start(2)] + new_block + text[bm.end(2):]

    if changed == 0:
        print("  all shifts already up to date")
    else:
        print(f"  {changed} shift(s) updated")

    if not dry_run:
        test_path.write_text(new_text)
    return True


# ── update bias_DFFs.sv ───────────────────────────────────────────────────────

def update_bias_sv(sv_path: Path, biases: list, dry_run: bool) -> bool:
    if len(biases) != TOTAL_BIASES:
        print(f"ERROR: bias.hex has {len(biases)} entries, expected {TOTAL_BIASES}")
        return False

    # Build the new case body
    case_lines = []
    for layer_name, bias_off, n_ch in BIAS_LAYERS:
        case_lines.append(f"            // {layer_name} (bias_off={bias_off}, {n_ch} channels)")
        for i in range(n_ch):
            idx = bias_off + i
            u   = int(biases[idx]) & 0xFFFFFFFF
            case_lines.append(f"            9'd{idx}: data = 32'sh{u:08X};")
        case_lines.append("")
    case_lines.append("            default: data = 32'sh00000000;")
    new_case_body = "\n".join(case_lines)

    sv_text = sv_path.read_text()
    case_re = re.compile(
        r'(always @\(\*\) begin\s*\n\s*case \(addr\)\s*\n)'
        r'(.*?)'
        r'(\n\s*endcase)',
        re.DOTALL,
    )
    m = case_re.search(sv_text)
    if not m:
        print("ERROR: could not locate case statement in bias_DFFs.sv")
        return False

    new_sv = sv_text[:m.start(2)] + new_case_body + sv_text[m.end(2):]

    if not dry_run:
        sv_path.write_text(new_sv)
    print(f"  {TOTAL_BIASES} bias entries written")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", type=Path, default=MODEL_DIR,
                        help="Directory containing scales.txt and bias.hex (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing files")
    args = parser.parse_args()

    scales_path = args.ckpt_dir / "scales.txt"
    bias_path   = args.ckpt_dir / "bias.hex"

    missing = [p for p in [scales_path, bias_path, TEST_FILE, BIAS_SV] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: not found: {p}")
        sys.exit(1)

    print(f"scales.txt : {scales_path}")
    print(f"bias.hex   : {bias_path}")
    print(f"test file  : {TEST_FILE}")
    print(f"bias sv    : {BIAS_SV}")
    if args.dry_run:
        print("(dry run — no files written)\n")
    else:
        print()

    shifts = load_shifts(scales_path)
    biases = load_biases(bias_path)
    print(f"Shifts ({len(shifts)}): {shifts}")
    print()

    ok = True

    print("LAYER_CFGS shifts: loaded dynamically from scales.txt by rtl_golden.load_layer_cfgs() — no patching needed.")

    print("\nUpdating bias_DFFs.sv case statement...")
    ok &= update_bias_sv(BIAS_SV, biases, args.dry_run)

    if not ok:
        sys.exit(1)

    if not args.dry_run:
        print("\nDone. Next:")
        print("  cd src/rtl/dscnn/kws_top")
        print("  python3 generate_spect.py --keyword yes")
        print("  make test-cocotb")


if __name__ == "__main__":
    main()
