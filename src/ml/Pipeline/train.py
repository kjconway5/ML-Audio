import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "dscnn"))
from dscnn import create_model
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ── Power-of-2 QAT ────────────────────────────────────────────────────────────
# The RTL applies one arithmetic right-shift per layer to requantize INT32 → INT8.
# A right-shift by N is equivalent to multiplying by 2^-N — an exact power of 2.
# Standard fbgemm QAT learns arbitrary floating-point per-channel scales, which
# export.py then rounds to the nearest power of 2, introducing requantization error.
#
# This observer snaps the scale to the nearest power of 2 DURING training so the
# model learns weights that are robust to exactly the arithmetic the RTL uses.
# Weights also use per-tensor (not per-channel) quantization so the effective
# output scale s_in * s_w is constant across all channels in a layer — matching
# the single shift the RTL applies to every output channel equally.

class PowerOf2Observer(torch.quantization.observer.MinMaxObserver):
    """MinMaxObserver that snaps the computed scale to the nearest power of 2."""
    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        log2_scale   = torch.log2(scale.clamp(min=1e-8))
        snapped_scale = torch.pow(2.0, torch.round(log2_scale))
        return snapped_scale, zero_point


def get_pow2_qat_qconfig():
    """
    QAT qconfig with power-of-2 scales and per-tensor weight quantization.
    Produces scales that map exactly to the RTL's integer right-shift — no
    rounding error at export time.
    """
    act_fq = torch.quantization.FakeQuantize.with_args(
        observer=PowerOf2Observer.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=255,
        ),
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    )
    wt_fq = torch.quantization.FakeQuantize.with_args(
        observer=PowerOf2Observer.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            quant_min=-128,
            quant_max=127,
        ),
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-128,
        quant_max=127,
    )
    return torch.quantization.QConfig(activation=act_fq, weight=wt_fq)
# ──────────────────────────────────────────────────────────────────────────────


class NpyDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloaders(config):
    output_dir = config["data"]["output_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)

    train_ds = NpyDataset(f"{output_dir}/train_features.npy", f"{output_dir}/train_labels.npy")
    val_ds   = NpyDataset(f"{output_dir}/val_features.npy",   f"{output_dir}/val_labels.npy")
    test_ds  = NpyDataset(f"{output_dir}/test_features.npy",  f"{output_dir}/test_labels.npy")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def disable_observer(model):
    for module in model.modules():
        if hasattr(module, 'apply_fake_quant'):
            # For FakeQuantize modules
            if hasattr(module, 'disable_observer'):
                module.disable_observer()
        if hasattr(module, 'observer_enabled'):
            module.observer_enabled[0] = 0


def migrate_optimizer_to_cpu(optimizer):
    for param_state in optimizer.state.values():
        for k, v in param_state.items():
            if isinstance(v, torch.Tensor):
                param_state[k] = v.cpu()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d} [Train]")

    for batch, labels in progress_bar:
        batch  = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch.size(0)

        current_acc = 100.0 * correct / total
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

    return total_loss / total, 100.0 * correct / total


def validate(model, val_loader, criterion, device, epoch, mode="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch:2d} [{mode:>4s}]")

    with torch.no_grad():
        for batch, labels in progress_bar:
            batch  = batch.to(device)
            labels = labels.to(device)

            outputs = model(batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch.size(0)

            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

    return total_loss / total, 100.0 * correct / total


def main():
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    gpu_available = torch.cuda.is_available()
    device     = torch.device("cuda" if gpu_available else "cpu")
    cpu_device = torch.device("cpu")

    print(f"\nFloat warmup device  : {device}")
    print(f"QAT / convert device : {cpu_device}  (PyTorch quantization requires CPU)")


    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)          

    criterion = nn.CrossEntropyLoss()

    train_cfg = config["training"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    lr_schedule = train_cfg["lr_schedule"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_schedule["milestones"],
        gamma=lr_schedule["gamma"],
    )

    n_epochs  = train_cfg["n_epochs"]
    val_every = train_cfg.get("val_every", 5)

   
    qat_start_epoch = train_cfg.get("qat_start_epoch", max(1, int(n_epochs * 0.7)))

    
    freeze_bn_epoch = train_cfg.get("freeze_bn_epoch", n_epochs - 2)

    
    qat_backend = "pow2"  # always power-of-2; config value ignored

    print(f"\nTraining configuration:")
    print(f"  - Total epochs        : {n_epochs}")
    print(f"  - Batch size          : {train_cfg['batch_size']}")
    print(f"  - Initial LR          : {train_cfg['learning_rate']}")
    print(f"  - Float warmup range  : epochs 1 – {qat_start_epoch - 1}  on {device}")
    print(f"  - QAT range           : epochs {qat_start_epoch} – {n_epochs}  on {cpu_device}")
    print(f"  - BN freeze epoch     : {freeze_bn_epoch}")
    print(f"  - QAT backend         : {qat_backend}")

    log_file = config["output"].get("log_file", "training_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"QAT Training started at {datetime.now()}\n\n")

    print("\n" + "="*70)
    print(f" Phase 1: Float32 Warmup  (epochs 1 – {qat_start_epoch - 1})")
    print("="*70 + "\n")

    best_val_acc = 0.0
    qat_prepared = False

    for epoch in range(1, n_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{n_epochs} | LR: {current_lr:.6f} | Device: {device}")
        print("-" * 70)

        
        if epoch == qat_start_epoch and not qat_prepared:

            print(f"\n{'='*70}")
            print(f" Phase 2: Quantization-Aware Training  (epochs {qat_start_epoch} – {n_epochs})")
            print(f"{'='*70}")

            if gpu_available:
                print(f"\n>>> Migrating model and optimizer: {device} → {cpu_device} <<<\n")

                
                model.to(cpu_device)

                
                migrate_optimizer_to_cpu(optimizer)

                print(f"  → Model and optimizer state now on {cpu_device}.")

            
            device = cpu_device
            print(f"  → `device` updated to {device}; all future batches go to CPU.\n")

            print("  → Fusing Conv+BN+ReLU layers...")
            model.eval()
            model.fuse_model()
            model.train()  # Back to train mode for QAT

            # ── QAT qconfig selection ──────────────────────────────────────────
            # Option A — Power-of-2 QAT (matches RTL exactly, revert to Option B
            #            if this model is unneeded):
            model.qconfig = get_pow2_qat_qconfig()

            # Option B — Standard fbgemm QAT (per-channel float scales, higher
            #            accuracy but requires rounding at export → RTL mismatch):
            # model.qconfig = torch.quantization.get_default_qat_qconfig(qat_backend)
            # ───────────────────────────────────────────────────────────────────

            torch.quantization.prepare_qat(model, inplace=True)

            qat_prepared = True
            print("  → prepare_qat() complete: fake quantizers active.\n")
        
        if epoch == freeze_bn_epoch and qat_prepared:
            print(f"\n>>> Epoch {epoch}: Freezing BatchNorm statistics <<<")
            
            freeze_bn_stats(model)
            print("  → BN stats frozen.\n")
        
        if epoch == freeze_bn_epoch + 1 and qat_prepared:
            print(f"\n>>> Epoch {epoch}: Freezing quantizer observer ranges <<<")
            
            disable_observer(model)
            print("  → Observer ranges frozen.\n")
        

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        log_msg = f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        print(f"{'':11s}  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        if epoch % val_every == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, mode="Val")
            log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            print(f"{'':11s}  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"{'':11s}  *** New best validation accuracy! ***")

        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

        scheduler.step()

    
    print("\n" + "="*70)
    print(" Phase 3: INT8 Conversion")
    print("="*70)
    print("\n>>> Converting QAT model to INT8 <<<\n")
    model.eval()
    torch.quantization.convert(model, inplace=True)
    print("  → Conversion complete: model is now INT8.\n")
    

    print("\n" + "="*70)
    print(" Final Test Evaluation  (INT8 model on CPU)")
    print("="*70 + "\n")

    test_loss, test_acc = validate(model, test_loader, criterion, device, n_epochs, mode="Test")
    print(f"\n{'':11s}  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Training completed at {datetime.now()}\n")
        f.write(f"Best validation accuracy  : {best_val_acc:.2f}%\n")
        f.write(f"Final INT8 test accuracy  : {test_acc:.2f}%\n")
        f.write(f"Final INT8 test loss      : {test_loss:.4f}\n")

    # Load labels from output/config.json (created by process_data.py)
    import json
    output_dir = config["data"]["output_dir"]
    labels_path = Path(output_dir) / "config.json"
    if labels_path.exists():
        with open(labels_path) as f:
            data_config = json.load(f)
        labels = data_config.get('labels', ['silence', 'unknown', 'yes'])
        label_to_id = data_config.get('label_to_id', {})
    else:
        labels = sorted(config.get('data', {}).get('classes', ['silence', 'unknown', 'yes']))
        label_to_id = {l: i for i, l in enumerate(labels)}

    save_path = config["output"]["model_save_path"]
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'labels': labels,  # Save correct label order!
        'label_to_id': label_to_id,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'quantized': True,
        'qat_backend': qat_backend,
    }, save_path)

    print(f"\n{'='*70}")
    print(f" Training + Quantization Complete!")
    print(f"{'='*70}")
    print(f"  Float warmup on      : {'GPU (CUDA)' if gpu_available else 'CPU (no GPU found)'}")
    print(f"  QAT + convert on     : CPU")
    print(f"  Best val accuracy    : {best_val_acc:.2f}%")
    print(f"  Final INT8 test acc  : {test_acc:.2f}%")
    print(f"  Model saved to       : {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
