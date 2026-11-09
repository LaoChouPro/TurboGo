import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
import sys

# Add project root to sys.path to allow importing 'src' modules when running from subdirectories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from src.dataset as requested (reuse dataset)
from src.dataset import GoDataset
# Import from src_large.model (use larger model)
from src_large.model import TinyGoNet

import argparse

def get_user_input(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        return type(default)(value)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default

def train_interactive():
    print("=== TinyGo Large Model Training Configuration ===")
    
    # Default values
    default_epochs = 5
    default_batch_size = 64
    default_lr = 0.001
    default_limit = 0 # 0 means all
    
    epochs = get_user_input("Epochs", default_epochs)
    batch_size = get_user_input("Batch Size", default_batch_size)
    lr = get_user_input("Learning Rate", default_lr)
    
    limit_input = input(f"Data Limit (0 for all, or N samples) [0]: ").strip()
    if not limit_input:
        limit = 0
    else:
        try:
            limit = int(limit_input)
        except:
            limit = 0
            
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\nStarting training with: Epochs={epochs}, Batch={batch_size}, LR={lr}, Limit={limit if limit > 0 else 'All'}, Device={device}")
    train(epochs=epochs, batch_size=batch_size, lr=lr, limit=limit, device=device)

def apply_gpu_augment(inputs, labels):
    """
    Apply random rotation and flip to the batch on GPU.
    """
    B = inputs.shape[0]
    SIZE = 19
    
    def rotate_coords(rows, cols, k):
        for _ in range(k):
            rows, cols = SIZE - 1 - cols, rows
        return rows, cols

    for i in range(1, 8):
        mask_idx = slice(i, B, 8)
        sub_inputs = inputs[mask_idx]
        if sub_inputs.shape[0] == 0:
            continue
            
        sub_labels = labels[mask_idx]
        k = i % 4
        flip = (i >= 4)
        
        if flip:
            sub_inputs = sub_inputs.flip(-2)
            
        if k > 0:
            sub_inputs = torch.rot90(sub_inputs, k, [-2, -1])
            
        inputs[mask_idx] = sub_inputs
        
        rows = sub_labels // SIZE
        cols = sub_labels % SIZE
        
        if flip:
            rows = SIZE - 1 - rows
            
        if k > 0:
            rows, cols = rotate_coords(rows, cols, k)
            
        labels[mask_idx] = rows * SIZE + cols
        
    return inputs, labels

def train(epochs=10, batch_size=64, lr=0.001, limit=0, device='cpu'):
    data_path = "processed_data.npz"
    if not os.path.exists(data_path):
        data_path = "example_processed.npz"
        if not os.path.exists(data_path):
             print("Data file not found. Please run preprocess.py first.")
             return
    
    print(f"Using device: {device}")
    
    # Load dataset
    # Limit handling is now inside GoDataset for efficient random sampling
    full_dataset = GoDataset(data_path, augment=True, limit=limit)
    
    # Split Train/Val
    print("Splitting dataset...")
    val_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print("Creating DataLoaders...")
    # Optimized for Windows + RTX 5070 Ti
    # num_workers=4 provides good balance without excessive overhead
    # pin_memory=True speeds up transfer to GPU
    # persistent_workers=True avoids process recreation overhead
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # Model - Using default large params (10 blocks, 128 channels)
    model = TinyGoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Add Scheduler: Reduce LR when Val Acc stops improving
    # verbose removed in newer pytorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Resume if exists - CHANGED to best_model_large.pth
    start_epoch = 0
    best_acc = 0.0
    model_save_path = "best_model_large.pth"
    
    if os.path.exists(model_save_path):
        print(f"Resuming from {model_save_path}...")
        state = torch.load(model_save_path, map_location=device)
        # Check if state is full checkpoint or just model
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            start_epoch = state['epoch']
            best_acc = state.get('acc', 0.0)
            
            # Load optimizer and scheduler if available
            if 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            if 'scheduler_state_dict' in state:
                scheduler.load_state_dict(state['scheduler_state_dict'])
                
            print(f"Resumed from epoch {start_epoch} with acc {best_acc:.2f}%")
        else:
            model.load_state_dict(state)
            print("Resumed from legacy model file (weights only).")
            
    # optimizer moved up
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        print(f"Epoch {epoch+1} starting iteration...")
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply GPU Augmentation
            if device != 'cpu':
                inputs, labels = apply_gpu_augment(inputs, labels)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_correct_top5 = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Top-1
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Top-5
                _, top5_pred = outputs.topk(5, 1, True, True)
                # Expand labels to (batch, 5) for comparison
                labels_expanded = labels.view(-1, 1).expand_as(top5_pred)
                val_correct_top5 += (top5_pred == labels_expanded).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_acc_top5 = 100 * val_correct_top5 / val_total
        
        # Update Scheduler
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] Summary: Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Top-5: {val_acc_top5:.2f}%, LR: {current_lr:.6f}, Time: {time.time()-start_time:.1f}s")
        
        # Save best model - CHANGED to best_model_large.pth
        if val_acc > best_acc:
            best_acc = val_acc
            # Save full checkpoint with optimizer/scheduler state
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'acc': best_acc
            }, model_save_path)
            print(f"Saved new best model (Acc: {best_acc:.2f}%).")

    print("Training finished.")

if __name__ == "__main__":
    train_interactive()
