#!/usr/bin/env python3
"""Offline reference for full_demo WAV classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from host_common import (  # noqa: E402
    DEFAULT_MODEL,
    ML_ROOT,
    N_FRAMES,
    SAMPLES_STREAM,
    START_FRAME,
    wav_samples,
)

sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "my_test"))
from golden_model import GoldenExtractor, load_input_quant, quant_q610_to_int8  # noqa: E402


def load_dscnn(model_dir: Path):
    from dscnn import DSCNN

    ckpt = torch.load(model_dir / f"{model_dir.name}.pt",
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    labels = (ckpt.get("labels")
              or cfg.get("data", {}).get("classes")
              or ["no", "off", "on", "silence", "unknown", "wow", "yes"])

    model_cfg = cfg.get("model", {})
    feat_cfg = (ckpt.get("pipeline") or cfg.get("pipeline", {})
                ).get("feature_extractor", {}) or cfg.get("preprocessing", {})

    model = DSCNN(
        n_classes=len(labels),
        n_mels=feat_cfg.get("n_mels", 40),
        first_conv_filters=model_cfg.get("first_conv", {}).get("filters", 32),
        first_conv_kernel=tuple(model_cfg.get("first_conv", {}).get("kernel_size", [10, 4])),
        first_conv_stride=tuple(model_cfg.get("first_conv", {}).get("stride", [2, 2])),
        n_ds_blocks=model_cfg.get("ds_blocks", {}).get("n_blocks", 4),
        ds_filters=model_cfg.get("ds_blocks", {}).get("filters", 32),
        ds_kernel=tuple(model_cfg.get("ds_blocks", {}).get("kernel_size", [3, 3])),
        ds_stride=tuple(model_cfg.get("ds_blocks", {}).get("stride", [1, 1])),
    )

    state_keys = list(ckpt["model_state_dict"].keys())
    has_packed = any("_packed_params" in key for key in state_keys)
    has_qat = any("activation_post_process" in key for key in state_keys)

    if ckpt.get("quantized") or has_packed or has_qat:
        backend = ckpt.get("qat_backend", "fbgemm")
        int8_sym = backend == "qnnpack_int8sym"
        engine = "qnnpack" if int8_sym or backend == "qnnpack" else (
            backend if backend in ("fbgemm", "x86", "onednn") else "fbgemm")
        torch.backends.quantized.engine = engine
        model.eval()
        model.fuse_model()

        if ckpt.get("quantized") or has_packed:
            if int8_sym:
                observer = torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                model.qconfig = torch.quantization.QConfig(
                    activation=observer, weight=observer)
            else:
                model.qconfig = torch.quantization.get_default_qconfig(engine)
            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)
        else:
            model.train()
            if int8_sym:
                observer = torch.quantization.MovingAverageMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                fake_quant = torch.quantization.FakeQuantize.with_args(
                    observer=observer,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    quant_min=-128,
                    quant_max=127,
                )
                model.qconfig = torch.quantization.QConfig(
                    activation=fake_quant, weight=fake_quant)
            else:
                model.qconfig = torch.quantization.get_default_qat_qconfig(engine)
            torch.quantization.prepare_qat(model, inplace=True)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, labels


def predict_one(path: Path, model, labels, extractor, input_scale: float,
                input_mult: int, input_shift: int):
    samples = wav_samples(path, n_required=SAMPLES_STREAM)
    feats_q610 = extractor.extract(samples).T
    window = feats_q610[START_FRAME:START_FRAME + N_FRAMES]
    int8_spect = quant_q610_to_int8(window, input_mult, input_shift)
    tensor = torch.from_numpy(
        (int8_spect.astype(np.float32) * input_scale)
    ).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        idx = int(probs.argmax(1).item())

    top3 = torch.topk(probs[0], 3)
    return idx, [(int(i), labels[int(i)], float(p))
                 for p, i in zip(top3.values, top3.indices)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model directory under src/ml/models")
    parser.add_argument("wavs", nargs="*",
                        help="WAV files; defaults to src/ml/my_test/test_wavs/*.wav")
    args = parser.parse_args()

    model_dir = ML_ROOT / "models" / args.model
    if not model_dir.exists():
        print(f"missing model dir: {model_dir}")
        return 1

    input_scale, _spect_shift, input_mult, input_shift = load_input_quant(model_dir)
    model, labels = load_dscnn(model_dir)
    extractor = GoldenExtractor()

    wavs = [Path(wav) for wav in args.wavs]
    if not wavs:
        wavs = sorted((ML_ROOT / "my_test" / "test_wavs").glob("*.wav"))

    print(f"model={args.model} labels={labels}")
    for wav in wavs:
        try:
            idx, top3 = predict_one(wav, model, labels, extractor,
                                    input_scale, input_mult, input_shift)
            top3_s = ", ".join(f"{label}({prob:.2f})" for _, label, prob in top3)
            print(f"  {wav.name:20s} -> {labels[idx]:8s} class={idx} top3: {top3_s}")
        except Exception as exc:
            print(f"  {wav.name:20s} -> ERROR: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
