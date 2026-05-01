#!/usr/bin/env python3
"""
update_rtl.py - Retarget DS-CNN RTL artifacts to an exported model.

This script is intentionally the handoff point between training/export and RTL.
It supports the current 24-filter and 32-filter DS-CNN variants:

  - copies weights.hex into rtl/dscnn/weight_sram/
  - regenerates bias_dff/bias_DFFs.sv from bias.hex
  - retargets feature_sram.sv and weight_sram.sv depth/bank counts
  - retargets KWS/chip-core testbenches to the selected model directory

Usage:
    python3 update_rtl.py --ckpt ../models/dscnn-24full-v1
    python3 update_rtl.py --ckpt ../models/dscnn-32requant-v11 --dry-run

If the checkpoint is not trained yet, --filters can be used to preview the RTL
memory retargeting, but weights.hex/bias.hex/scales.txt are still required for
a real non-dry-run update.
"""

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # Pipeline -> ml -> src -> repo root
MODELS_DIR = HERE.parent / "models"

BIAS_SV = REPO_ROOT / "src/rtl/dscnn/bias_dff/bias_DFFs.sv"
FEATURE_SRAM_SV = REPO_ROOT / "src/rtl/dscnn/feature_sram/feature_sram.sv"
WEIGHT_SRAM_SV = REPO_ROOT / "src/rtl/dscnn/weight_sram/weight_sram.sv"
WEIGHT_SRAM_DIR = REPO_ROOT / "src/rtl/dscnn/weight_sram"
KWS_TEST_PY = REPO_ROOT / "src/rtl/dscnn/kws_top/test_kws_top.py"
CHIP_CORE_TB_PY = REPO_ROOT / "cocotb/chip_core_tb.py"

SUPPORTED_FILTERS = (24, 32)
N_CLASSES = 7
N_DS_BLOCKS = 4
FIRST_KH = 10
FIRST_KW = 4
DS_KH = 3
DS_KW = 3
OFMAP_H = 25
OFMAP_W = 20
SRAM_BANK_DEPTH = 1024


@dataclass(frozen=True)
class Layer:
    name: str
    layer: int
    in_ch: int
    out_ch: int
    kH: int
    kW: int
    sh: int
    sw: int
    ph: int
    pw: int
    dw: int
    w_off: int
    relu: int
    ofmap_h: int
    ofmap_w: int
    bias_off: int

    @property
    def n_weights(self) -> int:
        return self.out_ch * self.kH * self.kW if self.dw else self.out_ch * self.in_ch * self.kH * self.kW

    @property
    def n_biases(self) -> int:
        return self.out_ch


@dataclass(frozen=True)
class Arch:
    filters: int
    layers: list[Layer]
    weight_count: int
    bias_count: int
    feature_depth: int
    feature_banks: int
    feature_addr_w: int
    weight_banks: int
    weight_addr_w: int


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def addr_width(depth: int) -> int:
    return max(1, (depth - 1).bit_length())


def build_arch(filters: int) -> Arch:
    if filters not in SUPPORTED_FILTERS:
        raise ValueError(f"filters={filters} is not supported; expected one of {SUPPORTED_FILTERS}")

    layers = []
    w_off = 0
    bias_off = 0

    def add(name, layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, relu):
        nonlocal w_off, bias_off
        l = Layer(name, layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
                  dw, w_off, relu, OFMAP_H, OFMAP_W, bias_off)
        layers.append(l)
        w_off += l.n_weights
        bias_off += l.n_biases

    add("first_conv", 0, 1, filters, FIRST_KH, FIRST_KW, 2, 2, 4, 1, 0, 1)
    for i in range(N_DS_BLOCKS):
        add(f"ds_blocks.{i}.depthwise", 1 + i * 2, 1, filters, DS_KH, DS_KW, 1, 1, 1, 1, 1, 1)
        add(f"ds_blocks.{i}.pointwise", 2 + i * 2, filters, filters, 1, 1, 1, 1, 0, 0, 0, 1)
    add("classifier", 9, filters, N_CLASSES, 1, 1, 1, 1, 0, 0, 0, 0)

    feature_depth = filters * OFMAP_H * OFMAP_W
    feature_banks = ceil_div(feature_depth, SRAM_BANK_DEPTH)
    weight_banks = ceil_div(w_off, SRAM_BANK_DEPTH)

    return Arch(
        filters=filters,
        layers=layers,
        weight_count=w_off,
        bias_count=bias_off,
        feature_depth=feature_depth,
        feature_banks=feature_banks,
        feature_addr_w=addr_width(feature_banks * SRAM_BANK_DEPTH),
        weight_banks=weight_banks,
        weight_addr_w=addr_width(weight_banks * SRAM_BANK_DEPTH),
    )


def resolve_model_dir(arg: Path | None, allow_missing: bool = False) -> Path:
    if arg is None:
        candidates = []
        for d in MODELS_DIR.iterdir():
            m = re.fullmatch(r"dscnn-(24|32).*-v(\d+)", d.name)
            if d.is_dir() and m:
                candidates.append((int(m.group(2)), d.stat().st_mtime, d))
        if not candidates:
            raise FileNotFoundError(f"No dscnn-24/32 model directories found in {MODELS_DIR}")
        return max(candidates)[2]

    if arg.suffix == ".pt":
        arg = arg.parent
    if arg.exists():
        return arg
    alt = MODELS_DIR / arg
    if alt.exists():
        return alt
    if allow_missing:
        return alt if not arg.is_absolute() and len(arg.parts) == 1 else arg
    raise FileNotFoundError(f"model directory not found: {arg}")


def model_rel_from_ml(model_dir: Path) -> str:
    try:
        return model_dir.resolve().relative_to((REPO_ROOT / "src/ml").resolve()).as_posix()
    except ValueError:
        return f"models/{model_dir.name}"


def infer_filters(model_dir: Path, override: int | None) -> int:
    if override is not None:
        return override

    m = re.search(r"dscnn-(24|32)", model_dir.name)
    if m:
        return int(m.group(1))

    config_path = model_dir / "config.yaml"
    if config_path.exists():
        text = config_path.read_text()
        m = re.search(r"ds_blocks:\s*(?:\n\s+.*)*?\n\s+filters:\s*(\d+)", text)
        if m:
            return int(m.group(1))

    raise ValueError(
        f"could not infer filter count from {model_dir}; pass --filters 24 or --filters 32"
    )


def load_biases(bias_path: Path) -> list[int]:
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


def count_data_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip() and not line.lstrip().startswith("//"))


def replace_text(path: Path, new_text: str, dry_run: bool) -> bool:
    old_text = path.read_text()
    changed = old_text != new_text
    if changed and not dry_run:
        path.write_text(new_text)
    return changed


def update_bias_sv(arch: Arch, bias_path: Path, dry_run: bool) -> bool:
    biases = load_biases(bias_path)
    if len(biases) != arch.bias_count:
        print(f"ERROR: bias.hex has {len(biases)} entries, expected {arch.bias_count} for {arch.filters} filters")
        return False

    lines = []
    for l in arch.layers:
        lines.append(f"            // {l.name} (bias_off={l.bias_off}, {l.n_biases} channels)")
        for i in range(l.n_biases):
            idx = l.bias_off + i
            lines.append(f"            9'd{idx}: data = 32'sh{int(biases[idx]) & 0xFFFFFFFF:08X};")
        lines.append("")
    lines.append("            default: data = 32'sh00000000;")
    body = "\n".join(lines)

    text = BIAS_SV.read_text()
    text = re.sub(
        r"parameter DEPTH\s*=\s*\d+\s*,\s*//.*",
        f"parameter DEPTH  = {arch.bias_count},   // 9x{arch.filters} channels + 7 classifier = {arch.bias_count} ({arch.filters}-filter model)",
        text,
        count=1,
    )
    text = re.sub(
        r"parameter ADDR_W\s*=\s*\d+\s*//.*",
        f"parameter ADDR_W = 9      // 9-bit: supports bias offsets up to {arch.layers[-1].bias_off}",
        text,
        count=1,
    )

    new_text, n = re.subn(
        r"(always @\(\*\) begin\s*\n\s*case \(addr\)\s*\n).*?(\n\s*endcase)",
        r"\1" + body + r"\2",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        print(f"ERROR: could not locate case statement in {BIAS_SV}")
        return False

    changed = replace_text(BIAS_SV, new_text, dry_run)
    print(f"  {'would update' if dry_run and changed else 'updated' if changed else 'already current'} {arch.bias_count} bias entries")
    return True


def update_feature_sram(arch: Arch, dry_run: bool) -> bool:
    text = FEATURE_SRAM_SV.read_text()
    text = re.sub(r"// Ping-pong  feature-map buffer: 2 banks [x×] [\d,]+ [x×] 8-bit INT8",
                  f"// Ping-pong  feature-map buffer: 2 banks x {arch.feature_depth:,} x 8-bit INT8", text)
    text = re.sub(r"// Each bank implemented as \d+[x×] cascaded sram1024x8 macros",
                  f"// Each bank implemented as {arch.feature_banks}x cascaded sram1024x8 macros", text)
    text = re.sub(r"// \(\d+ [x×] 1024 = [\d,]+ capacity; valid range 0[-–][\d,]+\)",
                  f"// ({arch.feature_banks} x 1024 = {arch.feature_banks * SRAM_BANK_DEPTH:,} capacity; valid range 0-{arch.feature_depth - 1:,})", text)
    text = re.sub(r"parameter DEPTH\s*=\s*\d+",
                  f"parameter DEPTH  = {arch.feature_depth}", text)
    text = re.sub(r"parameter ADDR_W\s*=\s*\d+\s*// covers 0[-–]\d+; valid range 0[-–][\d,]+",
                  f"parameter ADDR_W = {arch.feature_addr_w}   // covers 0-{(1 << arch.feature_addr_w) - 1}; valid range 0-{arch.feature_depth - 1}", text)
    text = re.sub(r"// \d+ x 1024 = [\d,]+",
                  f"// {arch.feature_banks} x 1024 = {arch.feature_banks * SRAM_BANK_DEPTH:,}", text)
    text = re.sub(r"localparam NUM_BANKS = \d+;",
                  f"localparam NUM_BANKS = {arch.feature_banks};", text)

    changed = replace_text(FEATURE_SRAM_SV, text, dry_run)
    print(f"  feature_sram: depth={arch.feature_depth}, banks={arch.feature_banks}, ADDR_W={arch.feature_addr_w} ({'would update' if dry_run and changed else 'updated' if changed else 'already current'})")
    return True


def update_weight_sram(arch: Arch, dry_run: bool) -> bool:
    text = WEIGHT_SRAM_SV.read_text()
    text = re.sub(r"// Weight storage with write port for Subservient and read port for FSM: [\d,]+ [x×] 8-bit INT8 values",
                  f"// Weight storage with write port for Subservient and read port for FSM: {arch.weight_count:,} x 8-bit INT8 values", text)
    text = re.sub(r"// Implemented as \d+[x×] cascaded gf180mcu_ocd_ip_sram__sram1024x8m8wm1 macros",
                  f"// Implemented as {arch.weight_banks}x cascaded gf180mcu_ocd_ip_sram__sram1024x8m8wm1 macros", text)
    text = re.sub(r"// \(\d+ [x×] 1024 = \d+\)",
                  f"// ({arch.weight_banks} x 1024 = {arch.weight_banks * SRAM_BANK_DEPTH})", text)
    text = re.sub(r"parameter DEPTH\s*=\s*\d+",
                  f"parameter DEPTH  = {arch.weight_count}", text)
    text = re.sub(r"parameter ADDR_W\s*=\s*\d+\s*// covers 0[-–]\d+; valid range 0[-–]\d+",
                  f"parameter ADDR_W = {arch.weight_addr_w}   // covers 0-{(1 << arch.weight_addr_w) - 1}; valid range 0-{arch.weight_count - 1}", text)
    text = re.sub(r"localparam NUM_BANKS = \d+;",
                  f"localparam NUM_BANKS = {arch.weight_banks};", text)

    changed = replace_text(WEIGHT_SRAM_SV, text, dry_run)
    print(f"  weight_sram: depth={arch.weight_count}, banks={arch.weight_banks}, ADDR_W={arch.weight_addr_w} ({'would update' if dry_run and changed else 'updated' if changed else 'already current'})")
    return True


def update_chip_core_tb(arch: Arch, model_dir: Path, dry_run: bool) -> bool:
    rel = model_rel_from_ml(model_dir)
    text = CHIP_CORE_TB_PY.read_text()
    text = re.sub(r'_MODEL_DIR = _ML / "models/[^"]+"',
                  f'_MODEL_DIR = _ML / "{rel}"', text)
    if "MODEL_FILTERS =" in text:
        text = re.sub(r"MODEL_FILTERS = \d+", f"MODEL_FILTERS = {arch.filters}", text)
    else:
        text = text.replace("SPECT_DIR     = _KWS_DIR   / \"spectrograms\"\n",
                            "SPECT_DIR     = _KWS_DIR   / \"spectrograms\"\n"
                            f"MODEL_FILTERS = {arch.filters}\n")
    text = re.sub(r"load_layer_cfgs\(SCALES_TXT(?:,\s*n_filters=\d+)?\)",
                  f"load_layer_cfgs(SCALES_TXT, n_filters={arch.filters})", text)

    changed = replace_text(CHIP_CORE_TB_PY, text, dry_run)
    print(f"  chip_core_tb.py: model={rel}, filters={arch.filters} ({'would update' if dry_run and changed else 'updated' if changed else 'already current'})")
    return True


def update_kws_test(arch: Arch, model_dir: Path, dry_run: bool) -> bool:
    rel_model = model_dir.name
    text = KWS_TEST_PY.read_text()
    text = re.sub(r'MODEL_DIR\s+= os\.path\.join\(test_dir, "\.\.", "\.\.", "\.\.", "ml", "models", "[^"]+"\)',
                  f'MODEL_DIR     = os.path.join(test_dir, "..", "..", "..", "ml", "models", "{rel_model}")',
                  text)
    text = re.sub(r"load_layer_cfgs\(Path\(scales_path\)(?:,\s*n_filters=\d+)?\)",
                  f"load_layer_cfgs(Path(scales_path), n_filters={arch.filters})", text)
    text = re.sub(r"assert len\(weights\) == \d+, f\"Expected \d+ weights, got \{len\(weights\)\}\"",
                  f"assert len(weights) == {arch.weight_count}, f\"Expected {arch.weight_count} weights, got {{len(weights)}}\"", text)

    changed = replace_text(KWS_TEST_PY, text, dry_run)
    print(f"  test_kws_top.py: model={rel_model}, filters={arch.filters}, weights={arch.weight_count} ({'would update' if dry_run and changed else 'updated' if changed else 'already current'})")
    return True


def copy_weights(weights_src: Path, dry_run: bool) -> bool:
    weights_dest = WEIGHT_SRAM_DIR / "weights.hex"
    if not dry_run:
        shutil.copy2(weights_src, weights_dest)
    print(f"  weights.hex {'would copy' if dry_run else 'copied'} -> {weights_dest}")
    return True


def validate_export_files(arch: Arch, model_dir: Path) -> tuple[Path, Path, Path, bool]:
    scales_path = model_dir / "scales.txt"
    bias_path = model_dir / "bias.hex"
    weights_path = model_dir / "weights.hex"

    ok = True
    for p in (scales_path, bias_path, weights_path):
        if not p.exists():
            print(f"ERROR: not found: {p}")
            ok = False

    if ok:
        n_weights = count_data_lines(weights_path)
        n_biases = count_data_lines(bias_path)
        n_scales = count_data_lines(scales_path) - 2  # header + separator
        if n_weights != arch.weight_count:
            print(f"ERROR: weights.hex has {n_weights} values, expected {arch.weight_count}")
            ok = False
        if n_biases != arch.bias_count:
            print(f"ERROR: bias.hex has {n_biases} values, expected {arch.bias_count}")
            ok = False
        if n_scales != len(arch.layers):
            print(f"ERROR: scales.txt has {n_scales} layer rows, expected {len(arch.layers)}")
            ok = False

    return scales_path, bias_path, weights_path, ok


def print_arch_summary(arch: Arch, model_dir: Path) -> None:
    print(f"model dir       : {model_dir}")
    print(f"filters         : {arch.filters}")
    print(f"weights         : {arch.weight_count} ({arch.weight_banks} x 1024 SRAM banks)")
    print(f"biases          : {arch.bias_count}")
    print(f"feature depth   : {arch.feature_depth} per bank ({arch.feature_banks} x 1024 macros per bank)")
    print()
    print("Layer layout:")
    for l in arch.layers:
        print(f"  L{l.layer}: {l.name:24s} in={l.in_ch:2d} out={l.out_ch:2d} "
              f"w_off={l.w_off:5d} n_w={l.n_weights:4d} bias_off={l.bias_off:3d}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path,
                        help="Model directory or checkpoint file. Defaults to latest dscnn-24/32 model dir.")
    parser.add_argument("--filters", type=int, choices=SUPPORTED_FILTERS,
                        help="Override/inject filter count when the checkpoint is not available yet.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned changes without writing files.")
    args = parser.parse_args()

    try:
        model_dir = resolve_model_dir(args.ckpt, allow_missing=args.dry_run and args.filters is not None)
        filters = infer_filters(model_dir, args.filters)
        arch = build_arch(filters)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print_arch_summary(arch, model_dir)
    if args.dry_run:
        print("(dry run - no files written)\n")

    scales_path, bias_path, weights_path, files_ok = validate_export_files(arch, model_dir)
    required_sources = [BIAS_SV, FEATURE_SRAM_SV, WEIGHT_SRAM_SV, KWS_TEST_PY, CHIP_CORE_TB_PY]
    missing_sources = [p for p in required_sources if not p.exists()]
    if missing_sources:
        for p in missing_sources:
            print(f"ERROR: source file not found: {p}")
        files_ok = False

    if not files_ok and not args.dry_run:
        print("\nExport files are not ready. Train/export the model, then rerun update_rtl.py.")
        print("For a structural preview only, use --dry-run after creating the model directory/config.")
        sys.exit(1)

    ok = True
    print("Applying RTL/testbench updates:")
    if files_ok:
        ok &= copy_weights(weights_path, args.dry_run)
        ok &= update_bias_sv(arch, bias_path, args.dry_run)
    else:
        print("  skipping weights/bias updates because exported files are not present")
    ok &= update_feature_sram(arch, args.dry_run)
    ok &= update_weight_sram(arch, args.dry_run)
    ok &= update_chip_core_tb(arch, model_dir, args.dry_run)
    ok &= update_kws_test(arch, model_dir, args.dry_run)

    if not ok:
        sys.exit(1)

    print("\nDone.")
    print("Next:")
    print("  1. Regenerate KWS spectrogram vectors for this checkpoint.")
    print("  2. Run the KWS/chip simulations from your terminal.")


if __name__ == "__main__":
    main()
