import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from model import create_model
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NpyDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # (N, T, M)
        self.labels = np.load(labels_path)       # (N,)

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
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,       # Cross Entropy Loss function (CEP)
    optimizer: optim.Optimizer, # SGD 
    device: torch.device,
    epoch: int,
) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d} [Train]")
    
    for batch, labels in progress_bar:      # Load batches of audio, labels = classes (yes, silence, unknown)
        batch = batch.to(device)            #move data to GPU if possible 
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()       # reset gradients from previous iteration 
        outputs = model(batch)      # set outputs 
        loss = criterion(outputs, labels)   # Compute how wrong predictions are (CEP)

        # Backward pass
        loss.backward() # Backpropogation, compute gradients of loss based on paramters 
        optimizer.step() # Update model weights using ^ new gradients 

        # Track metrics
        total_loss += loss.item() * batch.size(0)   # Compute average loss with respect to different batch sizes (.item tensor -> number)
        _, predicted = outputs.max(1)   # Gets predicted class for each sample 
        correct += predicted.eq(labels).sum().item() # Compares predicted samples to actual labels to find # correct 
        total += batch.size(0)
        
        # Update progress bar
        current_acc = 100.0 * correct / total   # Find % of correct predictions
        progress_bar.set_postfix({      # Update progress bar 
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# Same as Training except without any gradient/weight changes. Only validation no training 
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,   
    device: torch.device,
    epoch: int,
    mode: str = "Val"
) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch:2d} [{mode:>4s}]")

    with torch.no_grad():
        for batch, labels in progress_bar:
            batch = batch.to(device)
            labels = labels.to(device)

            outputs = model(batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch.size(0)
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    # Load configuration from yaml file 
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: SGD with momentum
    train_cfg = config["training"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )
    
    # Learning rate scheduler
    lr_schedule = train_cfg["lr_schedule"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_schedule["milestones"],
        gamma=lr_schedule["gamma"],
    )
    
    # Training settings
    n_epochs = train_cfg["n_epochs"]
    val_every = train_cfg.get("val_every", 5)
    
    print(f"\nTraining configuration:")
    print(f"  - Epochs: {n_epochs}")
    print(f"  - Batch size: {train_cfg['batch_size']}")
    print(f"  - Initial LR: {train_cfg['learning_rate']}")
    print(f"  - Momentum: {train_cfg['momentum']}")
    print(f"  - LR decay at epochs: {lr_schedule['milestones']}")
    print(f"  - Validation every {val_every} epochs")
    
    # Setup logging
    log_file = config["output"].get("log_file", "training_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        # f.write(f"Model parameters: {n_params:,}\n")
        f.write(f"Configuration: {config_path}\n\n")
    
    print("\n" + "="*70)
    print(" Starting Training")
    print("="*70 + "\n")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{n_epochs} | LR: {current_lr:.6f}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        log_msg = f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        print(f"{'':11s}  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate at specified epochs
        if epoch % val_every == 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, mode="Val"
            )
            
            log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            print(f"{'':11s}  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
            
            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"{'':11s}  *** New best validation accuracy! ***")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # Update learning rate
        scheduler.step()
    
    # Final test evaluation at epoch 20
    print("\n" + "="*70)
    print(" Final Test Evaluation")
    print("="*70 + "\n")
    
    test_loss, test_acc = validate(
        model, test_loader, criterion, device, n_epochs, mode="Test"
    )
    
    print(f"\n{'':11s}  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Log final results
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Training completed at {datetime.now()}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final test accuracy: {test_acc:.2f}%\n")
        f.write(f"Final test loss: {test_loss:.4f}\n")
    
    # Save final model
    save_path = config["output"]["model_save_path"]
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
    }, save_path)
    
    print(f"\n{'='*70}")
    print(f" Training Complete!")
    print(f"{'='*70}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final test accuracy: {test_acc:.2f}%")
    print(f"  Model saved to: {save_path}")
    print(f"  Training log saved to: {log_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()