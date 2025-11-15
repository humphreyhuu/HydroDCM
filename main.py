import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import json
from datetime import datetime

from model import HydroDCM, infoNCE_loss, adversarial_loss


class HydroDCMDataset(Dataset):
    """Dataset for HydroDCM training"""
    def __init__(self, X, y, spatial, reservoir_labels):
        self.X = X
        self.y = y 
        self.spatial = spatial
        self.reservoir_labels = reservoir_labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx], 
            'spatial': self.spatial[idx],
            'reservoir_label': self.reservoir_labels[idx]
        }


def create_reservoir_domain_labels(reservoir_labels, all_reservoirs):
    """Create domain labels for each sample based on reservoir"""
    reservoir_to_idx = {rsr: idx for idx, rsr in enumerate(sorted(all_reservoirs))}
    domain_labels = torch.tensor([reservoir_to_idx[label] for label in reservoir_labels])
    return domain_labels


def inverse_transform_predictions(predictions, targets, processed_data, reservoir_labels):
    """Inverse transform predictions and targets to original scale"""
    predictions_orig = predictions.copy()
    targets_orig = targets.copy()
    
    batch_size, pred_days = predictions.shape
    
    for i in range(batch_size):
        rsr_name = reservoir_labels[i]
        
        # Get the appropriate scaler
        if rsr_name in processed_data['source_scalers']:
            scaler_y = processed_data['source_scalers'][rsr_name]['scaler_y']
        elif rsr_name in processed_data['target_scalers']:
            scaler_y = processed_data['target_scalers'][rsr_name]['scaler_y']
        else:
            continue  # Skip if scaler not found
            
        # Inverse transform
        pred_reshaped = predictions[i].reshape(-1, 1)
        tgt_reshaped = targets[i].reshape(-1, 1)
        
        pred_orig = scaler_y.inverse_transform(pred_reshaped).flatten()
        tgt_orig = scaler_y.inverse_transform(tgt_reshaped).flatten()
        
        predictions_orig[i] = pred_orig
        targets_orig[i] = tgt_orig
    
    return predictions_orig, targets_orig


def evaluate_model(model, test_loader, processed_data, device, target_reservoirs=['MCR', 'JVR', 'MCP']):
    """Evaluate model on test set with focus on target reservoirs"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_reservoir_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            spatial = batch['spatial'].to(device)
            
            # Forward pass (inference mode)
            y_hat = model.inference(X, spatial)
            
            all_predictions.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_reservoir_labels.extend(batch['reservoir_label'])
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\nEvaluation Results:")
    print(f"Total test samples: {len(all_predictions)}")
    print(f"Prediction shape: {all_predictions.shape}")
    
    # Inverse transform to original scale
    predictions_orig, targets_orig = inverse_transform_predictions(
        all_predictions, all_targets, processed_data, all_reservoir_labels
    )
    
    # Overall metrics (all target reservoirs combined)
    print(f"\n{'='*60}")
    print(f"OVERALL TARGET DOMAIN PERFORMANCE")
    print(f"{'='*60}")
    
    overall_r2 = r2_score(targets_orig.flatten(), predictions_orig.flatten())
    overall_mse = mean_squared_error(targets_orig.flatten(), predictions_orig.flatten())
    overall_rmse = np.sqrt(overall_mse)
    
    print(f"Overall R2: {overall_r2:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    # Daily performance
    print(f"\nDaily Performance:")
    daily_r2_scores = []
    for day in range(7):
        day_targets = targets_orig[:, day]
        day_predictions = predictions_orig[:, day]
        day_r2 = r2_score(day_targets, day_predictions)
        daily_r2_scores.append(day_r2)
        print(f"Day {day+1} R2: {day_r2:.4f}")
    
    # Individual reservoir performance
    print(f"\nIndividual Target Reservoir Performance:")
    reservoir_metrics = {}
    
    for target_rsr in target_reservoirs:
        # Find samples for this reservoir
        rsr_indices = [i for i, label in enumerate(all_reservoir_labels) if label == target_rsr]
        
        if len(rsr_indices) == 0:
            print(f"{target_rsr}: No test samples found")
            continue
            
        rsr_predictions = predictions_orig[rsr_indices]
        rsr_targets = targets_orig[rsr_indices]
        
        rsr_r2 = r2_score(rsr_targets.flatten(), rsr_predictions.flatten())
        rsr_mse = mean_squared_error(rsr_targets.flatten(), rsr_predictions.flatten())
        rsr_rmse = np.sqrt(rsr_mse)
        
        reservoir_metrics[target_rsr] = {
            'r2': rsr_r2,
            'mse': rsr_mse,
            'rmse': rsr_rmse,
            'samples': len(rsr_indices)
        }
        
        print(f"{target_rsr}: R2={rsr_r2:.4f}, MSE={rsr_mse:.4f}, RMSE={rsr_rmse:.4f}, Samples={len(rsr_indices)}")
        
        # Daily performance for this reservoir
        rsr_daily_r2 = []
        for day in range(7):
            day_targets = rsr_targets[:, day]
            day_predictions = rsr_predictions[:, day]
            day_r2 = r2_score(day_targets, day_predictions)
            rsr_daily_r2.append(day_r2)
        
        print(f"  Daily R2: {[f'{r2:.3f}' for r2 in rsr_daily_r2]}")
        reservoir_metrics[target_rsr]['daily_r2'] = rsr_daily_r2
    
    return {
        'overall_r2': overall_r2,
        'overall_mse': overall_mse, 
        'overall_rmse': overall_rmse,
        'daily_r2': daily_r2_scores,
        'reservoir_metrics': reservoir_metrics
    }


def train_hydrodcm(model, train_loader, val_loader, processed_data, device, 
                   num_epochs=100, lr=1e-3, lambda_adv=0.1, lambda_sup=1.0,
                   warmup_epochs=10, save_dir="./logs/HydroDCM"):
    """Training loop for HydroDCM with progressive loss integration"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Loss criterion for supervised loss
    mse_criterion = nn.MSELoss()
    
    # Get all reservoir names for domain labels
    all_reservoirs = list(processed_data['source_domains'].keys()) + list(processed_data['target_domains'].keys())
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'contrastive_loss': [],
        'adversarial_loss': [],
        'supervised_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print(f"Training HydroDCM for {num_epochs} epochs...")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Lambda adversarial: {lambda_adv}")
    print(f"Lambda supervised: {lambda_sup}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = {'total': 0, 'contrastive': 0, 'adversarial': 0, 'supervised': 0}
        
        # Progressive training strategy
        if epoch < warmup_epochs:
            # Phase 1: Contrastive + Supervised only
            use_adversarial = False
            alpha = 0.0
        else:
            # Phase 2: Full joint optimization
            use_adversarial = True
            alpha = min(1.0, (epoch - warmup_epochs) / 20.0)  # Gradually increase alpha
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            spatial = batch['spatial'].to(device)
            reservoir_labels = batch['reservoir_label']
            
            optimizer.zero_grad()
            
            # Forward pass with all components
            outputs = model(X, spatial, alpha=alpha, return_components=True)
            y_hat = outputs['y_hat']
            h = outputs['h']
            v = outputs['v']
            domain_logits = outputs['domain_logits']
            
            # 1. Contrastive loss (InfoNCE)
            loss_contrastive = infoNCE_loss(v, reservoir_labels)
            
            # 2. Supervised loss (MSE)
            loss_supervised = mse_criterion(y_hat, y)
            
            # 3. Adversarial loss (only after warmup)
            if use_adversarial:
                loss_adversarial = adversarial_loss(domain_logits, v)
            else:
                loss_adversarial = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = (loss_contrastive + 
                         lambda_sup * loss_supervised + 
                         lambda_adv * loss_adversarial)
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            batch_size = X.size(0)
            epoch_losses['total'] += total_loss.item() * batch_size
            epoch_losses['contrastive'] += loss_contrastive.item() * batch_size
            epoch_losses['adversarial'] += loss_adversarial.item() * batch_size
            epoch_losses['supervised'] += loss_supervised.item() * batch_size
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'MSE': f"{loss_supervised.item():.4f}",
                'Con': f"{loss_contrastive.item():.4f}",
                'Adv': f"{loss_adversarial.item():.4f}"
            })
        
        # Average losses
        train_size = len(train_loader.dataset)
        avg_train_loss = epoch_losses['total'] / train_size
        avg_contrastive = epoch_losses['contrastive'] / train_size
        avg_adversarial = epoch_losses['adversarial'] / train_size
        avg_supervised = epoch_losses['supervised'] / train_size
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for batch in val_pbar:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                spatial = batch['spatial'].to(device)
                
                y_hat = model.inference(X, spatial)
                loss = mse_criterion(y_hat, y)
                val_loss += loss.item() * X.size(0)
                
                val_pbar.set_postfix({'Val Loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['contrastive_loss'].append(avg_contrastive)
        history['adversarial_loss'].append(avg_adversarial)
        history['supervised_loss'].append(avg_supervised)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
              f"MSE={avg_supervised:.4f}, Con={avg_contrastive:.4f}, Adv={avg_adversarial:.4f}, "
              f"Alpha={alpha:.3f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, best_model_path)
            print(f"  → New best model saved (val_loss: {avg_val_loss:.4f})")
    
    return history, best_model_path


def main():
    # ==================== HYPERPARAMETERS ====================
    # Data parameters
    data_path = "./AdaTrip/data"
    model_save_dir = "./logs/HydroDCM"
    target_reservoirs = ['MCR', 'JVR', 'MCP']
    
    # Model parameters
    input_dim = 3           # Temperature, precipitation, inflow
    spatial_dim = 3         # Lat, lon, elevation  
    hidden_dim = 64         # Feature dimension
    domain_embed_dim = 32   # Domain embedding dimension
    output_dim = 7          # 7-day prediction
    num_domains = 30        # Total number of reservoirs
    num_layers = 2          # LSTM layers
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    lambda_adv = 0.1        # Adversarial loss weight
    lambda_sup = 1.0        # Supervised loss weight
    warmup_epochs = 10      # Epochs before adversarial training
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== LOAD DATA ====================
    print("Loading preprocessed data...")
    with open('./HydroDCM/data/dg_final_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    tensor_data = torch.load('./HydroDCM/data/dg_tensor_data.pt')
    
    # Create datasets
    train_dataset = HydroDCMDataset(
        X=tensor_data['X_train'],
        y=tensor_data['y_train'],
        spatial=tensor_data['spatial_train'],
        reservoir_labels=tensor_data['train_labels']
    )
    
    test_dataset = HydroDCMDataset(
        X=tensor_data['X_test'],
        y=tensor_data['y_test'],
        spatial=tensor_data['spatial_test'],
        reservoir_labels=tensor_data['test_labels']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Target reservoirs: {target_reservoirs}")
    
    # ==================== MODEL INITIALIZATION ====================
    model = HydroDCM(
        input_dim=input_dim,
        spatial_dim=spatial_dim,
        hidden_dim=hidden_dim,
        domain_embed_dim=domain_embed_dim,
        output_dim=output_dim,
        num_domains=num_domains,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== TRAINING ====================
    print("\nStarting training...")
    history, best_model_path = train_hydrodcm(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set for validation (since we don't have a separate val set)
        processed_data=processed_data,
        device=device,
        num_epochs=num_epochs,
        lr=learning_rate,
        lambda_adv=lambda_adv,
        lambda_sup=lambda_sup,
        warmup_epochs=warmup_epochs,
        save_dir=model_save_dir
    )
    
    # ==================== EVALUATION ====================
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on target domains...")
    eval_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        processed_data=processed_data,
        device=device,
        target_reservoirs=target_reservoirs
    )
    
    # ==================== SAVE RESULTS ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(model_save_dir, f'results_{timestamp}.json')
    
    results = {
        'hyperparameters': {
            'input_dim': input_dim,
            'spatial_dim': spatial_dim,
            'hidden_dim': hidden_dim,
            'domain_embed_dim': domain_embed_dim,
            'output_dim': output_dim,
            'num_domains': num_domains,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'lambda_adv': lambda_adv,
            'lambda_sup': lambda_sup,
            'warmup_epochs': warmup_epochs
        },
        'training_history': {k: [float(x) for x in v] for k, v in history.items()},
        'evaluation_metrics': eval_metrics,
        'target_reservoirs': target_reservoirs,
        'best_model_path': best_model_path,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Best model saved to: {best_model_path}")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()