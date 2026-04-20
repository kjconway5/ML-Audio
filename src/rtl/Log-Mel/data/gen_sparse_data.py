#!/usr/bin/env python3
"""
Generate sparse mel filterbank data files for the optimised SRAM layout.

Reads the original mel_coeffs.hex, mel_starts.hex, mel_ends.hex and produces:
  - mel_coeffs_sparse.hex : coefficients packed densely (no zero-padding)
  - mel_indices.hex       : combined index SRAM [starts|ends|offsets]

Index SRAM layout (256 x 8-bit):
  [0:39]   = start_bin for each filter
  [40:79]  = end_bin   for each filter
  [80:119] = coeff_base_offset for each filter (index into sparse coeff array)
"""
import sys
from pathlib import Path


def main():
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(".")

    # Read original files
    with open(data_dir / "mel_starts.hex") as f:
        starts = [int(line.strip(), 16) for line in f if line.strip()]
    with open(data_dir / "mel_ends.hex") as f:
        ends = [int(line.strip(), 16) for line in f if line.strip()]
    with open(data_dir / "mel_coeffs.hex") as f:
        all_coeffs = [int(line.strip(), 16) for line in f if line.strip()]

    n_mels = len(starts)
    max_coeffs = 16  # original padding width

    assert len(starts) == n_mels
    assert len(ends) == n_mels
    assert len(all_coeffs) == n_mels * max_coeffs, \
        f"Expected {n_mels * max_coeffs} coeffs, got {len(all_coeffs)}"

    # Build sparse coefficients and offsets
    sparse_coeffs = []
    offsets = []

    for m in range(n_mels):
        n_bins = ends[m] - starts[m] + 1
        base = m * max_coeffs
        offsets.append(len(sparse_coeffs))

        # Take only the nonzero-range coefficients
        for i in range(n_bins):
            sparse_coeffs.append(all_coeffs[base + i])

    total = len(sparse_coeffs)
    print(f"Filters: {n_mels}")
    print(f"Original coeff count: {n_mels * max_coeffs}")
    print(f"Sparse coeff count:   {total}")
    print(f"Savings:              {n_mels * max_coeffs - total} entries "
          f"({100 * (n_mels * max_coeffs - total) / (n_mels * max_coeffs):.1f}%)")
    print(f"Max offset:           {max(offsets)} (fits in 8 bits: {max(offsets) < 256})")
    print()

    # Verify offsets fit in 8 bits
    assert max(offsets) < 256, f"Offset {max(offsets)} doesn't fit in 8 bits!"
    assert total <= 256, f"Total coeffs {total} doesn't fit in sram256!"

    # Write sparse coefficients (16-bit hex)
    out_coeffs = data_dir / "mel_coeffs_sparse.hex"
    with open(out_coeffs, "w") as f:
        for c in sparse_coeffs:
            f.write(f"{c:04x}\n")
    print(f"Wrote {len(sparse_coeffs)} entries to {out_coeffs}")

    # Write combined index SRAM (8-bit hex)
    # Layout: starts[0:39], ends[40:79], offsets[80:119]
    index_data = []
    for s in starts:
        index_data.append(s & 0xFF)
    for e in ends:
        index_data.append(e & 0xFF)
    for o in offsets:
        index_data.append(o & 0xFF)

    out_indices = data_dir / "mel_indices.hex"
    with open(out_indices, "w") as f:
        for val in index_data:
            f.write(f"{val:02x}\n")
    print(f"Wrote {len(index_data)} entries to {out_indices}")

    # Print summary table
    print("\nPer-filter details:")
    print(f"{'Filter':>6} {'Start':>5} {'End':>5} {'#Coeff':>6} {'Offset':>6}")
    for m in range(n_mels):
        n = ends[m] - starts[m] + 1
        print(f"{m:>6} {starts[m]:>5} {ends[m]:>5} {n:>6} {offsets[m]:>6}")


if __name__ == "__main__":
    main()