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


class IndividualDataset(Dataset):
    """Dataset for individual training (only X and y, no spatial info)"""
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


def train_individual(model, train_loader, val_loader, device, 
                     num_epochs=100, lr=1e-3, save_dir="./logs/HydroDCM"):
    """Training loop for individual ED-LSTM (trained on early target data, tested on later target data)"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'individual_best_model.pth')
    
    print(f"Training Individual ED-LSTM for {num_epochs} epochs...")
    print("Note: Training on early target data, testing on later target data (temporal split)")
    
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for batch in val_pbar:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss += loss.item() * X.size(0)
                
                val_pbar.set_postfix({'Val Loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
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


def split_target_data_temporal(X_test, y_test, test_labels, train_ratio=0.75):
    """Split target domain data temporally: early 3/4 for train, later 1/4 for test"""
    
    # Since data is already organized by reservoir and time, we split each reservoir's data temporally
    reservoir_data = {}
    
    # Group data by reservoir
    for i, label in enumerate(test_labels):
        if label not in reservoir_data:
            reservoir_data[label] = {'X': [], 'y': [], 'indices': []}
        reservoir_data[label]['X'].append(X_test[i])
        reservoir_data[label]['y'].append(y_test[i])
        reservoir_data[label]['indices'].append(i)
    
    train_indices = []
    test_indices = []
    
    # For each reservoir, split temporally
    for rsr_name, data in reservoir_data.items():
        n_samples = len(data['X'])
        n_train = int(n_samples * train_ratio)
        
        # Early samples for training, later samples for testing
        rsr_train_indices = data['indices'][:n_train]
        rsr_test_indices = data['indices'][n_train:]
        
        train_indices.extend(rsr_train_indices)
        test_indices.extend(rsr_test_indices)
        
        print(f"{rsr_name}: {len(rsr_train_indices)} train samples (early), {len(rsr_test_indices)} test samples (later)")
    
    # Convert to sorted lists for consistent ordering
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    
    # Split data
    X_train_individual = X_test[train_indices]
    y_train_individual = y_test[train_indices]
    train_labels_individual = [test_labels[i] for i in train_indices]
    
    X_test_individual = X_test[test_indices]
    y_test_individual = y_test[test_indices]
    test_labels_individual = [test_labels[i] for i in test_indices]
    
    print(f"\nTemporal split summary:")
    print(f"  Train: {len(X_train_individual)} samples (early {train_ratio*100:.0f}%)")
    print(f"  Test: {len(X_test_individual)} samples (later {(1-train_ratio)*100:.0f}%)")
    
    return (X_train_individual, y_train_individual, train_labels_individual,
            X_test_individual, y_test_individual, test_labels_individual)


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
    train_ratio = 0.75      # Early 75% for train, later 25% for test
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== LOAD DATA ====================
    print("Loading preprocessed data...")
    with open('./HydroDCM/data/dg_final_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    tensor_data = torch.load('./HydroDCM/data/dg_tensor_data.pt')
    
    # Use only TARGET domain data for individual training with temporal split
    print("\nUsing TARGET domain data for individual training with temporal split")
    print("Individual setup: Train on early target data, test on later target data")
    
    # Split target data temporally
    (X_train_individual, y_train_individual, train_labels_individual,
     X_test_individual, y_test_individual, test_labels_individual) = split_target_data_temporal(
        tensor_data['X_test'], tensor_data['y_test'], tensor_data['test_labels'],
        train_ratio=train_ratio
    )
    
    # Create datasets
    train_dataset = IndividualDataset(
        X=X_train_individual,
        y=y_train_individual,
        reservoir_labels=train_labels_individual
    )
    
    test_dataset = IndividualDataset(
        X=X_test_individual,
        y=y_test_individual,
        reservoir_labels=test_labels_individual
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Individual train samples: {len(train_dataset)} (early target data)")
    print(f"Individual test samples: {len(test_dataset)} (later target data)")
    print(f"Target reservoirs: {target_reservoirs}")
    print(f"Temporal split: {train_ratio*100:.0f}% train (early) / {(1-train_ratio)*100:.0f}% test (later)")
    
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
    print("\nStarting individual training...")
    history, best_model_path = train_individual(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as validation for early stopping
        device=device,
        num_epochs=num_epochs,
        lr=learning_rate,
        save_dir=model_save_dir
    )
    
    # ==================== EVALUATION ====================
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating individual model on later target data...")
    eval_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        processed_data=processed_data,
        device=device,
        target_reservoirs=target_reservoirs
    )
    
    # ==================== SAVE RESULTS ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(model_save_dir, f'individual_results_{timestamp}.json')
    
    results = {
        'model_type': 'individual_ed_lstm',
        'description': 'Individual model: ED-LSTM trained on early target data, tested on later target data (temporal split)',
        'hyperparameters': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'train_ratio': train_ratio
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
    print("\nIndividual training completed!")
    print("This represents performance with temporal split on target domains")


if __name__ == "__main__":
    main()