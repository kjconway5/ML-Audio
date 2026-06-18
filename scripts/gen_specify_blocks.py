#!/usr/bin/env python3
"""
gen_specify_blocks.py – Patch cell-library Verilog models with specify blocks
so that Icarus Verilog's $sdf_annotate can back-annotate post-PnR delays.

How values work
---------------
The specify block declares that a timing *path exists* in a cell module.
Every path in the specify block starts with a delay of 0.  At t=0, cocotb's
$sdf_annotate call replaces each cell *instance's* path delays with the
actual post-PnR numbers from the LibreLane SDF — real silicon-characterised
values from OpenROAD STA.  The zeros are never "seen" by the simulation.

The script reads the SDF to discover path topology:
  - IOPATH (combinational delay and clk→Q)
  - SETUP / HOLD / WIDTH timing checks

Usage
-----
    python3 gen_specify_blocks.py  <corner.sdf>  <cells.v>  <cells_patched.v>

The output file is a copy of <cells.v> with a `specify … endspecify` block
inserted into every module whose name appears in the SDF.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CellTiming:
    # (from_pin, to_pin) → present (values written as 0, SDF overrides them)
    iopath: set[tuple[str, str]] = field(default_factory=set)
    # Each element: (check_type, arg1, arg2) – deduplicated across instances
    checks: set[tuple[str, str, str | None]] = field(default_factory=set)


# ── SDF scanner ──────────────────────────────────────────────────────────────

def _paren_delta(line: str) -> int:
    """Net open-minus-close paren count, ignoring string literals."""
    delta, in_str, esc = 0, False, False
    for ch in line:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '(':
                delta += 1
            elif ch == ')':
                delta -= 1
    return delta


def scan_sdf(sdf_path: Path) -> dict[str, CellTiming]:
    """
    Scan the SDF and return {celltype: CellTiming} with path topology.

    Only the existence of paths is recorded — delay values are intentionally
    ignored here; $sdf_annotate provides them at simulation time.
    """
    timings: dict[str, CellTiming] = {}

    # State
    depth = 0
    cell_start_depth: int | None = None
    celltype: str | None = None
    section: str | None = None      # 'delay' | 'check' | None
    section_start_depth: int | None = None

    for raw in sdf_path.open():
        line = raw.strip()
        if not line:
            continue

        delta = _paren_delta(line)

        if cell_start_depth is None:
            # Outside any CELL block
            if line.startswith('(CELL'):
                cell_start_depth = depth   # depth BEFORE this line
            depth += delta
            continue

        # ── Inside a CELL block ───────────────────────────────────────────

        # CELLTYPE line
        m = re.match(r'\(CELLTYPE\s+"([^"]+)"\)', line)
        if m:
            celltype = m.group(1)
            if celltype not in timings:
                timings[celltype] = CellTiming()
            depth += delta
            continue

        # Section opens
        if section is None:
            if line.startswith('(DELAY'):
                section = 'delay'
                section_start_depth = depth
                depth += delta
                continue
            if line.startswith('(TIMINGCHECK'):
                section = 'check'
                section_start_depth = depth
                depth += delta
                continue

        depth += delta

        # Section close (depth returned to level before section opened)
        if section is not None and depth <= section_start_depth:
            section = None
            section_start_depth = None
            # Fall through to check CELL close below

        # CELL close
        if depth <= cell_start_depth:
            cell_start_depth = None
            celltype = None
            section = None
            continue

        # Collect timing data
        if celltype is None:
            continue
        ct = timings[celltype]

        if section == 'delay':
            # (IOPATH from to (rise_triplet) [(fall_triplet)])
            # from/to can be bare pins or edge-qualified: (posedge CLK)
            m = re.match(r'\(IOPATH\s+(\((?:pos|neg)edge\s+\S+\)|\S+)\s+(\S+)', line)
            if m:
                ct.iopath.add((m.group(1), m.group(2)))

        elif section == 'check':
            # (SETUP (edge data) (edge ref) (triplet))
            # (HOLD  (edge data) (edge ref) (triplet))
            # (WIDTH (edge pin)  (triplet))
            m = re.match(r'\((SETUP|HOLD|RECOVERY|REMOVAL)\s+'
                         r'(\((?:pos|neg)edge\s+\S+\)|\S+)\s+'
                         r'(\((?:pos|neg)edge\s+\S+\)|\S+)', line)
            if m:
                ct.checks.add((m.group(1), m.group(2), m.group(3)))
                continue

            m = re.match(r'\((WIDTH|PERIOD)\s+(\((?:pos|neg)edge\s+\S+\)|\S+)', line)
            if m:
                ct.checks.add((m.group(1), m.group(2), None))

    return timings


# ── Specify block generator ───────────────────────────────────────────────────

def _edge(pin_expr: str) -> str:
    """
    Normalise an SDF pin expression to Verilog specify syntax.

    '(posedge CLK)' → 'posedge CLK'
    'A'             → 'A'
    """
    m = re.match(r'\((pos|neg)edge\s+(\S+)\)', pin_expr)
    if m:
        return f'{m.group(1)}edge {m.group(2)}'
    return pin_expr


def build_specify(ct: CellTiming) -> list[str]:
    """Return the lines (no newlines) that make up the specify block."""
    lines: list[str] = ['  specify']

    # IOPATH → Verilog path delay:  (from => to) = 0;
    for (src, dst) in sorted(ct.iopath):
        src_v = _edge(src)
        dst_v = _edge(dst)
        if ' ' in src_v:   # edge-qualified
            lines.append(f'    ({src_v} => {dst_v}) = 0;')
        else:
            lines.append(f'    ({src_v} => {dst_v}) = 0;')

    # Timing checks (ct.checks is already a set, no need to track seen)
    for entry in sorted(ct.checks):
        ctype, arg1, arg2 = entry

        if ctype in ('SETUP', 'RECOVERY'):
            # $setup(data_event, ref_event, limit)
            lines.append(f'    ${ctype.lower()}({_edge(arg1)}, {_edge(arg2)}, 0);')

        elif ctype in ('HOLD', 'REMOVAL'):
            # $hold(ref_event, data_event, limit)  ← arguments REVERSED vs SDF
            lines.append(f'    ${ctype.lower()}({_edge(arg2)}, {_edge(arg1)}, 0);')

        elif ctype in ('WIDTH', 'PERIOD') and arg2 is None:
            lines.append(f'    ${ctype.lower()}({_edge(arg1)}, 0);')

    lines.append('  endspecify')
    return lines


# ── Verilog patcher ───────────────────────────────────────────────────────────

_MODULE_RE = re.compile(r'^module\s+(\w+)\s*[(\s]')
_ENDMODULE_RE = re.compile(r'^endmodule\b')


def patch_verilog(
    src: Path,
    dst: Path,
    timings: dict[str, CellTiming],
) -> dict[str, int]:
    """
    Copy *src* to *dst*, inserting specify blocks before each `endmodule`
    whose module name appears in *timings*.

    Returns {module_name: paths_added}.
    """
    patched: dict[str, int] = {}
    current_module: str | None = None
    has_specify = False
    out_lines: list[str] = []

    for raw in src.open():
        line = raw.rstrip('\n')

        m = _MODULE_RE.match(line)
        if m:
            current_module = m.group(1)
            has_specify = False

        if 'specify' in line and current_module:
            has_specify = True

        if _ENDMODULE_RE.match(line) and current_module:
            ct = timings.get(current_module)
            if ct and not has_specify:
                specify_lines = build_specify(ct)
                out_lines.extend(ln + '\n' for ln in specify_lines)
                patched[current_module] = len(ct.iopath) + len(ct.checks)
            current_module = None
            has_specify = False

        out_lines.append(line + '\n')

    dst.write_text(''.join(out_lines), encoding='utf-8')
    return patched


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = (argv or sys.argv)[1:]
    if len(args) != 3:
        print(__doc__)
        print('Usage: gen_specify_blocks.py <corner.sdf> <cells.v> <cells_patched.v>')
        return 1

    sdf_path  = Path(args[0])
    src_path  = Path(args[1])
    dst_path  = Path(args[2])

    if not sdf_path.exists():
        print(f'ERROR: SDF not found: {sdf_path}', file=sys.stderr)
        return 2
    if not src_path.exists():
        print(f'ERROR: Verilog not found: {src_path}', file=sys.stderr)
        return 2

    print(f'Scanning SDF: {sdf_path}')
    timings = scan_sdf(sdf_path)
    # Remove the top-level chip_top entry – it only has INTERCONNECT, not cells
    timings.pop('chip_top', None)
    print(f'  {len(timings)} cell types found in SDF')

    print(f'Patching:     {src_path}  →  {dst_path}')
    patched = patch_verilog(src_path, dst_path, timings)

    if patched:
        print(f'  Patched {len(patched)} module(s):')
        for name, n in sorted(patched.items()):
            print(f'    {name}  ({n} paths/checks)')
    else:
        print('  No modules patched (none matched SDF cell types in this file)')

    sdf_only = set(timings) - set(patched)
    if sdf_only:
        print(f'  {len(sdf_only)} SDF cell type(s) not found in this Verilog file '
              f'(may be in a different library file):')
        for name in sorted(sdf_only):
            print(f'    {name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
