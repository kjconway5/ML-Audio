#!/usr/bin/env python3
""" Patch SDF so Icarus Verilog 13 can consume it without
fatal errors or VPI assertion crashes

Usage:
    sdf_icarus_fixer.py [--dry-run] [--verbose] <input.sdf> <output.sdf>
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# Compiled patterns

# Header triplets that omit the typical slot:  val1::val2  →  val1:val2:val2
# Covers both forms emitted by OpenROAD:
#   bare:    (VOLTAGE 3.300::3.300)
#   quoted:  (PROCESS "1.000::1.000")
_TRIPLET_RE = re.compile(
    r'\(\s*(VOLTAGE|TEMPERATURE|PROCESS)\s+'
    r'(?:"\s*([\-\d.]+)\s*::\s*([\-\d.]+)\s*"'   # quoted  (PROCESS)
    r'|([\-\d.]+)\s*::\s*([\-\d.]+))'             # bare    (VOLTAGE / TEMPERATURE)
    r'\s*\)'
)

# Zero-delay top-level clk/rst pad INTERCONNECT entries that Icarus cannot
# resolve even without escape sequences.
# Example:  (INTERCONNECT clk_PAD clk_pad.PAD ...)
_TOP_PAD_INTERCONNECT_RE = re.compile(
    r'^\s*\(INTERCONNECT\s+\S+_PAD(?:\[\d+\])?\s+\w+_pad\.PAD\s+'
)

# Any backslash escape that Icarus 13's SDF path parser rejects.
# Covers \[ \] \.  as found in OpenROAD output for pad / SRAM instances.
# NOTE: the original code only checked for \[ and \. — this regex also
# catches \], which is equally fatal to vvp.
_ESCAPED_PATH_RE = re.compile(r'\\[\[.\]]')


# Stats container

@dataclass
class PatchStats:
    triplets_fixed: int = 0
    interconnects_dropped: int = 0
    cells_dropped: int = 0
    # Populated only when verbose=True
    dropped_interconnect_lines: list[str] = field(default_factory=list)
    dropped_cell_instances: list[str] = field(default_factory=list)


# Triplet fix

def _fix_triplet(m: re.Match, stats: PatchStats) -> str:
    """Return the patched triplet string and increment *stats*."""
    stats.triplets_fixed += 1
    keyword = m.group(1)
    if m.group(2) is not None:       # quoted PROCESS "v1::v2"
        v1, v2 = m.group(2), m.group(3)
        return f'({keyword} "{v1}:{v2}:{v2}")'
    v1, v2 = m.group(4), m.group(5) # bare VOLTAGE / TEMPERATURE
    return f'({keyword} {v1}:{v2}:{v2})'



# INTERCONNECT filter

def _is_unannotatable_interconnect(line: str) -> bool:
    """
    Return True if *line* is an INTERCONNECT entry that will crash vvp.

    Two failure modes are caught:

    * Escaped brackets or dots in the source or destination path
      (``\\[``, ``\\]``, ``\\.``) — Icarus 13's path splitter rejects them.
    * Zero-delay top-level clk/rst pad entries that Icarus cannot bind even
      without any escape sequences.
    """
    if not line.lstrip().startswith('(INTERCONNECT'):
        return False
    return bool(
        _ESCAPED_PATH_RE.search(line)
        or _TOP_PAD_INTERCONNECT_RE.match(line)
    )


# Paren-depth tracker (string-literal aware)

def _paren_delta(line: str) -> int:
    """Net open-minus-close paren count, ignoring characters inside strings.

    Checking *escaped* before *in_string* ensures that a backslash at the
    very end of a string literal (e.g. ``\\"``) correctly consumes the next
    character without re-entering the string-boundary logic.
    """
    delta = 0
    in_string = False
    escaped = False
    for ch in line:
        # Consume the character after a backslash unconditionally.
        if escaped:
            escaped = False
            continue
        if in_string:
            if ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '(':
                delta += 1
            elif ch == ')':
                delta -= 1
    return delta


# CELL block helpers

def _instance_line(cell_lines: list[str]) -> str:
    """Return the raw ``(INSTANCE ...)`` line from a buffered CELL block, or ''."""
    for line in cell_lines:
        if line.lstrip().startswith('(INSTANCE'):
            return line.strip()
    return ''


def _cell_has_unannotatable_instance(cell_lines: list[str]) -> bool:
    """True if any INSTANCE path in the block contains an escape Icarus rejects."""
    return any(
        line.lstrip().startswith('(INSTANCE') and _ESCAPED_PATH_RE.search(line)
        for line in cell_lines
    )


def _flush_cell(
    cell_buf: list[str],
    stats: PatchStats,
    out: list[str],
    *,
    verbose: bool,
) -> None:
    """Emit or discard a completed CELL block based on its INSTANCE path."""
    if _cell_has_unannotatable_instance(cell_buf):
        stats.cells_dropped += 1
        if verbose:
            inst = _instance_line(cell_buf)
            stats.dropped_cell_instances.append(inst)
            log.debug('Dropped CELL block  INSTANCE: %s', inst)
    else:
        out.extend(cell_buf)


# Single-pass INTERCONNECT + CELL filter

def _filter_lines(
    lines: list[str],
    stats: PatchStats,
    *,
    verbose: bool,
) -> list[str]:
    """
    Filter SDF lines in a single pass.

    * Drops INTERCONNECT lines Icarus 13 cannot annotate.
    * Collects CELL blocks and drops those whose INSTANCE path contains a
      backslash escape that Icarus cannot parse.

    Combining both filters avoids iterating the (potentially very large)
    line list twice, as the original two-pass approach did.
    """
    out: list[str] = []
    cell_buf: list[str] = []
    depth = 0

    for line in lines:
        # Inside a buffered CELL block
        if cell_buf:
            cell_buf.append(line)
            depth += _paren_delta(line)
            if depth <= 0:
                _flush_cell(cell_buf, stats, out, verbose=verbose)
                cell_buf.clear()
                depth = 0
            continue

        # Start of a new CELL block
        if line.lstrip().startswith('(CELL'):
            cell_buf.append(line)
            depth = _paren_delta(line)
            if depth <= 0:              # entire CELL on one line (unusual)
                _flush_cell(cell_buf, stats, out, verbose=verbose)
                cell_buf.clear()
                depth = 0
            continue

        #Unannotatable INTERCONNECT
        if _is_unannotatable_interconnect(line):
            stats.interconnects_dropped += 1
            if verbose:
                stats.dropped_interconnect_lines.append(line.rstrip())
                log.debug('Dropped INTERCONNECT: %s', line.rstrip())
            continue

        out.append(line)

    # Preserve a truncated/malformed tail so downstream tools can diagnose it.
    if cell_buf:
        log.warning(
            'Unclosed CELL block at EOF (%d lines); preserving as-is.', len(cell_buf)
        )
        out.extend(cell_buf)

    return out


# Top-level patch entry point

def fix(
    src: Path,
    dst: Path,
    *,
    verbose: bool = False,
    dry_run: bool = False,
) -> PatchStats:
    """
    Read *src*, apply all three patches, and write the result to *dst*
    (skipped when *dry_run* is True).

    Returns a :class:`PatchStats` describing every change made.
    """
    text = src.read_text(encoding='utf-8', errors='replace')
    stats = PatchStats()

    # Pass 1 — fix header triplets (whole-text regex substitution).
    text = _TRIPLET_RE.sub(lambda m: _fix_triplet(m, stats), text)

    # Pass 2 — drop unannotatable INTERCONNECT lines and CELL blocks.
    lines = _filter_lines(text.splitlines(keepends=True), stats, verbose=verbose)

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(''.join(lines), encoding='utf-8')

    return stats


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_sdf',  type=Path, metavar='input.sdf')
    parser.add_argument('output_sdf', type=Path, metavar='output.sdf')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report what would change without writing any output',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Log each dropped INTERCONNECT line and CELL instance',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    src: Path = args.input_sdf
    dst: Path = args.output_sdf

    if not src.exists():
        log.error('Input SDF not found: %s', src)
        return 2

    if src.resolve() == dst.resolve():
        log.error(
            'Input and output paths resolve to the same file; '
            'refusing to overwrite in-place.'
        )
        return 3

    stats = fix(src, dst, verbose=args.verbose, dry_run=args.dry_run)

    log.info(
        '[sdf_fix] %s: fixed %d triplet(s), dropped %d INTERCONNECT line(s), '
        'dropped %d CELL block(s)%s  →  %s',
        src.name,
        stats.triplets_fixed,
        stats.interconnects_dropped,
        stats.cells_dropped,
        ' (dry run — nothing written)' if args.dry_run else '',
        dst,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
