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

from model import EDLSTMEncoder


class SimpleEDLSTM(nn.Module):
    """Simple ED-LSTM baseline without any domain generalization components"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=7, num_layers=2, dropout=0.1):
        super(SimpleEDLSTM, self).__init__()
        
        # Use the same feature extractor as HydroDCM
        self.f_phi = EDLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Simple prediction head (same architecture as HydroDCM's p_omega)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, X):
        """
        Args:
            X: [B, T_in, F_in] input sequences
        Returns:
            y_hat: [B, H] predictions
        """
        # Extract temporal features
        h = self.f_phi(X)  # [B, H_feat]
        
        # Direct prediction without any domain adaptation
        y_hat = self.predictor(h)  # [B, output_dim]
        
        return y_hat


class BaselineDataset(Dataset):
    """Dataset for baseline training (only X and y, no spatial info)"""
    def __init__(self, X, y, reservoir_labels):
        self.X = X
        self.y = y 
        self.reservoir_labels = reservoir_labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'reservoir_label': self.reservoir_labels[idx]
        }


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
            
            # Forward pass
            y_hat = model(X)
            
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


def train_baseline(model, train_loader, test_loader, processed_data, device, 
                   num_epochs=100, lr=1e-3, save_dir="./logs/HydroDCM"):
    """Training loop for baseline ED-LSTM without domain generalization"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    best_test_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'baseline_best_model.pth')
    
    print(f"Training Baseline ED-LSTM for {num_epochs} epochs...")
    print("Note: Training on SOURCE domains, testing on TARGET domains (zero-shot)")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_hat = model(X)
            loss = criterion(y_hat, y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)
            
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Test phase (using target domains as test set)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]", leave=False)
            for batch in test_pbar:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                
                y_hat = model(X)
                loss = criterion(y_hat, y)
                test_loss += loss.item() * X.size(0)
                
                test_pbar.set_postfix({'Test Loss': f"{loss.item():.4f}"})
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Test={avg_test_loss:.4f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': avg_test_loss,
                'history': history
            }, best_model_path)
            print(f"  → New best model saved (test_loss: {avg_test_loss:.4f})")
    
    return history, best_model_path


def main():
    # ==================== HYPERPARAMETERS ====================
    # Data parameters
    model_save_dir = "./logs/HydroDCM"
    target_reservoirs = ['MCR', 'JVR', 'MCP']
    
    # Model parameters (same as HydroDCM for fair comparison)
    input_dim = 3           # Temperature, precipitation, inflow
    hidden_dim = 64         # Feature dimension
    output_dim = 7          # 7-day prediction
    num_layers = 2          # LSTM layers
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== LOAD DATA ====================
    print("Loading preprocessed data...")
    with open('./HydroDCM/data/dg_final_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    tensor_data = torch.load('./HydroDCM/data/dg_tensor_data.pt')
    
    # Create datasets (no spatial information needed for baseline)
    train_dataset = BaselineDataset(
        X=tensor_data['X_train'],
        y=tensor_data['y_train'],
        reservoir_labels=tensor_data['train_labels']
    )
    
    test_dataset = BaselineDataset(
        X=tensor_data['X_test'],
        y=tensor_data['y_test'],
        reservoir_labels=tensor_data['test_labels']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)} (Source domains)")
    print(f"Test samples: {len(test_dataset)} (Target domains)")
    print(f"Target reservoirs: {target_reservoirs}")
    print("\nBaseline setup: Train on source domains, test on target domains (zero-shot transfer)")
    
    # ==================== MODEL INITIALIZATION ====================
    model = SimpleEDLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture: Same ED-LSTM backbone as HydroDCM (fair comparison)")
    
    # ==================== TRAINING ====================
    print("\nStarting baseline training...")
    history, best_model_path = train_baseline(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        processed_data=processed_data,
        device=device,
        num_epochs=num_epochs,
        lr=learning_rate,
        save_dir=model_save_dir
    )
    
    # ==================== EVALUATION ====================
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating baseline on target domains...")
    eval_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        processed_data=processed_data,
        device=device,
        target_reservoirs=target_reservoirs
    )
    
    # ==================== SAVE RESULTS ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(model_save_dir, f'baseline_results_{timestamp}.json')
    
    results = {
        'model_type': 'baseline_ed_lstm',
        'description': 'Lower bound: ED-LSTM trained on source domains, tested on target domains (zero-shot)',
        'hyperparameters': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
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
    print("\nBaseline training completed!")
    print("This represents the LOWER BOUND performance (no domain adaptation)")


if __name__ == "__main__":
    main()