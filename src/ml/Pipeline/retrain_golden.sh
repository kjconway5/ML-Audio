#!/bin/bash
# retrain_golden.sh — Retrain DSCNN with golden (RTL-accurate) features.
#
# PREREQUISITE: Update config.yaml paths before running:
#   dataset.data_dir  → path to your Speech Commands dataset
#   data.output_dir   → where to save processed features
#   output.model_save_path → e.g., "dscnn-golden-retrained.pt"
#
# Usage:
#   cd src/ml/Pipeline
#   bash retrain_golden.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Step 1: Generate golden features (process_data_golden.py)"
echo "============================================================"
python3 process_data_golden.py

echo ""
echo "============================================================"
echo "  Step 2: Verify feature range is [0, ~16] (golden Q4.12)"
echo "============================================================"
python3 -c "
import numpy as np, yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
out = cfg['data']['output_dir']
for split in ['train', 'val', 'test']:
    f = np.load(f'{out}/{split}_features.npy')
    print(f'{split}: shape={f.shape}, range=[{f.min():.3f}, {f.max():.3f}], mean={f.mean():.3f}')
    assert f.max() < 17, f'ERROR: {split} features max={f.max():.2f} exceeds golden range [0,16]!'
    assert f.min() >= 0, f'ERROR: {split} features have negative values!'
print('All features in golden range [0, ~16]. OK.')
"

echo ""
echo "============================================================"
echo "  Step 3: Train DSCNN with golden features"
echo "============================================================"
python3 train.py

echo ""
echo "============================================================"
echo "  Step 4: Validate — quant.scale should match golden range"
echo "============================================================"
python3 -c "
import torch, yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
ckpt = torch.load(cfg['output']['model_save_path'], map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
q_scale = sd['quant.scale'].item()
obs_max = q_scale * 255
print(f'quant.scale = {q_scale:.6f}')
print(f'Observed range: [0, {obs_max:.2f}]')
assert obs_max < 25, f'ERROR: quant range {obs_max:.1f} too wide for golden features!'
assert obs_max > 10, f'ERROR: quant range {obs_max:.1f} too narrow!'
print(f'Quant range consistent with golden features. OK.')
# Check depthwise weight shape
dw = sd['ds_blocks.0.depthwise.weight']
print(f'Depthwise weight shape: {dw.shape}')
assert dw.shape[1] == 1, 'ERROR: depthwise groups!=in_channels. Check Pipeline/model.py!'
print('Architecture verified (true depthwise separable). OK.')
"

echo ""
echo "============================================================"
echo "  DONE. Run inference validation:"
echo "  python ../my_test/diagnosis_full.py --model <new_model.pt>"
echo "============================================================"
