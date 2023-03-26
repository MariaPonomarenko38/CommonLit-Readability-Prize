import torch

def train_loop(model, train_loader, criterion, optimizer, device):
    total_loss, total_batches = 0, 0
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
    return total_loss, total_batches

def val_loop(model, val_loader, criterion, device):
    total_val_loss, total_val_rmse, total_val_batches = 0, 0, 0
    model.eval()
    for batch in val_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
    
        total_val_loss += loss.item()
        total_val_rmse += torch.sqrt(loss).item()
        total_val_batches += 1
    return total_val_loss, total_val_rmse, total_val_batches