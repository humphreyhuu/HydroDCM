import numpy as np
import os
import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import torch
import geopandas as gpd


def load_spatial_data(data_path="./AdaTrip/data"):
    """Load spatial information for reservoirs (LAT, LON, ELEV)
    Args:
        data_path (str): Path to the data directory
    Returns:
        dict: Dictionary mapping reservoir names to spatial attributes
    """
    # Reservoir information from AdaTrip
    reservoir_info = {
        "BSR": 1990, "CAU": 1999, "CRY": 1982, "DCR": 1987, "DIL": 1985,
        "ECH": 1982, "ECR": 1992, "FGR": 1982, "FON": 1990, "GMR": 1982,
        "HYR": 1999, "JOR": 1997, "JVR": 1996, "LCR": 1998, "LEM": 1982,
        "MCP": 1991, "MCR": 1998, "NAV": 1986, "PIN": 1990, "RFR": 1989,
        "RID": 1990, "ROC": 1982, "RUE": 1982, "SCO": 1996, "SJR": 1992,
        "STA": 1982, "STE": 1982, "TPR": 1982, "USR": 1991, "VAL": 1986,
    }
    
    # Load spatial data from shapefile
    map_path = os.path.join(data_path, "map", "ReservoirElevations.shp")
    map_data = gpd.read_file(map_path)
    map_data = map_data[['Initials', 'Lat', 'Lon', 'RASTERVALU']]
    
    # Handle TPR mapping (TAY -> TPR)
    map_data.loc[map_data['Initials'] == 'TAY', 'Initials'] = 'TPR'
    
    # Filter for our reservoirs
    rsrs = list(reservoir_info.keys())
    map_data = map_data[map_data['Initials'].isin(rsrs)]
    map_data.reset_index(drop=True, inplace=True)
    map_data.columns = ['RSR', 'LAT', 'LON', 'ELEV']
    
    # Create spatial dictionary
    spatial_dict = {}
    for _, row in map_data.iterrows():
        rsr_name = row['RSR']
        spatial_dict[rsr_name] = {
            'LAT': row['LAT'],
            'LON': row['LON'], 
            'ELEV': row['ELEV']
        }
    
    return spatial_dict


def create_sliding_windows(data, _days_x=30, _days_y=7, _input_features=3):
    """Create sliding windows for X and y data for DG
    Args:
        data (np.array): Input data array
        _days_x (int): Number of input days (default 30)
        _days_y (int): Number of prediction days (default 7 for multi-step ahead)
        _input_features (int): Number of input features (temperature, precipitation, inflow)
    Returns:
        tuple: (X, y) where X is the input windows and y is the target values
    """
    X, y = [], []
    for i in range(len(data) - _days_x - _days_y + 1):
        # Input window: _days_x days of input features (T, P, Q)
        X.append(data[i:i+_days_x, :_input_features])
        # Target window: _days_y days of inflow
        y.append(data[i+_days_x:i+_days_x+_days_y, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_reservoir_data(data_path, reservoir_name, time_range=('1999-01-01', '2011-12-31'), _input_features=3, _days_x=30, _days_y=7):
    """Load reservoir data for specified time range
    Args:
        data_path (str): Path to the data directory
        reservoir_name (str): Name of the reservoir CSV file
        time_range (tuple): Start and end dates for data extraction
        _input_features (int): Number of input features
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
    Returns:
        dict: Dictionary containing X and y data
    """
    df = pd.read_csv(os.path.join(data_path, reservoir_name))
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for specified time range (1999-2011)
    start_date, end_date = time_range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    
    # Extract features (include inflow_y exclude date)
    features = df.iloc[:, :_input_features+1].values.astype(np.float32)
    filtered_data = features[mask]
    
    if len(filtered_data) == 0:
        return {'X': np.array([]), 'y': np.array([])}
    
    # Create sliding windows
    X, y = create_sliding_windows(filtered_data, _days_x, _days_y, _input_features)
    
    return {'X': X, 'y': y}


def preprocess_dg_data(data_path="./AdaTrip/data", output_path="./HydroDCM/data", 
                       _input_features=3, _days_x=30, _days_y=7, 
                       target_reservoirs=['MCR', 'JVR', 'MCP']):
    """Preprocess reservoir data for Domain Generalization
    Args:
        data_path (str): Path to the input data directory
        output_path (str): Path to save processed data
        _input_features (int): Number of input features
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
        target_reservoirs (list): List of target reservoir names for testing
    Returns:
        dict: Dictionary containing processed data split by domains
    """
    os.makedirs(output_path, exist_ok=True)
    align_path = os.path.join(data_path, "align")
    reservoir_files = [f for f in os.listdir(align_path) if f.endswith('.csv')]
    
    print("Loading reservoir data for Domain Generalization...")
    print(f"Time range: 1999-01-01 to 2011-12-31")
    print(f"Target reservoirs (test domains): {target_reservoirs}")
    
    # Load spatial information
    print("Loading spatial information...")
    spatial_data = load_spatial_data(data_path)
    
    source_data = {}  # Source domains (27 reservoirs for train/val)
    target_data = {}  # Target domains (3 reservoirs for test)
    
    # Load and categorize reservoirs by domain
    for rsr_file in reservoir_files:
        rsr_name = rsr_file.split('.')[0]
        data = load_reservoir_data(align_path, rsr_file, 
                                 time_range=('1999-01-01', '2011-12-31'),
                                 _input_features=_input_features, 
                                 _days_x=_days_x, _days_y=_days_y)
        
        if len(data['X']) == 0:
            print(f"Warning: No data found for reservoir {rsr_name} in specified time range")
            continue
            
        if rsr_name in target_reservoirs:
            target_data[rsr_name] = data
            print(f"Target domain: {rsr_name} - {len(data['X'])} samples")
        else:
            source_data[rsr_name] = data
            print(f"Source domain: {rsr_name} - {len(data['X'])} samples")
    
    print(f"\nDomain split summary:")
    print(f"Source domains: {len(source_data)} reservoirs")
    print(f"Target domains: {len(target_data)} reservoirs")
    
    # Apply local normalization (per reservoir) and add spatial info
    print("\nApplying local normalization per reservoir and adding spatial information...")
    
    # Normalize source domains
    source_scalers = {}
    normalized_source_data = {}
    spatial_scalers = {}
    
    for rsr_name, data in source_data.items():
        if len(data['X']) == 0:
            continue
            
        # Create individual scalers for each reservoir
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on this reservoir's training data
        X_reshaped = data['X'].reshape(-1, _input_features)
        y_reshaped = data['y'].reshape(-1, 1)
        
        scaler_X.fit(X_reshaped)
        scaler_y.fit(y_reshaped)
        
        # Transform data
        X_normalized = scaler_X.transform(X_reshaped).reshape(data['X'].shape)
        y_normalized = scaler_y.transform(y_reshaped).reshape(data['y'].shape)
        
        # Add spatial information 
        spatial_attr = spatial_data.get(rsr_name, {'LAT': 0.0, 'LON': 0.0, 'ELEV': 0.0})
        spatial_vector = np.array([spatial_attr['LAT'], spatial_attr['LON'], spatial_attr['ELEV']], dtype=np.float32)
        
        normalized_source_data[rsr_name] = {
            'X': X_normalized,
            'y': y_normalized,
            'spatial': spatial_vector
        }
        
        source_scalers[rsr_name] = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
    
    # Normalize target domains
    target_scalers = {}
    normalized_target_data = {}
    
    for rsr_name, data in target_data.items():
        if len(data['X']) == 0:
            continue
            
        # Create individual scalers for each reservoir
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on this reservoir's training data
        X_reshaped = data['X'].reshape(-1, _input_features)
        y_reshaped = data['y'].reshape(-1, 1)
        
        scaler_X.fit(X_reshaped)
        scaler_y.fit(y_reshaped)
        
        # Transform data
        X_normalized = scaler_X.transform(X_reshaped).reshape(data['X'].shape)
        y_normalized = scaler_y.transform(y_reshaped).reshape(data['y'].shape)
        
        # Add spatial information 
        spatial_attr = spatial_data.get(rsr_name, {'LAT': 0.0, 'LON': 0.0, 'ELEV': 0.0})
        spatial_vector = np.array([spatial_attr['LAT'], spatial_attr['LON'], spatial_attr['ELEV']], dtype=np.float32)
        
        normalized_target_data[rsr_name] = {
            'X': X_normalized,
            'y': y_normalized,
            'spatial': spatial_vector
        }
        
        target_scalers[rsr_name] = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
    
    # Standardize spatial features across all reservoirs
    print("Standardizing spatial features...")
    all_spatial_data = []
    for rsr_name in list(normalized_source_data.keys()) + list(normalized_target_data.keys()):
        if rsr_name in normalized_source_data:
            all_spatial_data.append(normalized_source_data[rsr_name]['spatial'])
        else:
            all_spatial_data.append(normalized_target_data[rsr_name]['spatial'])
    
    all_spatial_data = np.array(all_spatial_data)
    spatial_scaler = MinMaxScaler(feature_range=(0, 1))
    spatial_scaler.fit(all_spatial_data)
    
    # Apply spatial normalization
    for rsr_name in normalized_source_data:
        normalized_source_data[rsr_name]['spatial'] = spatial_scaler.transform(
            normalized_source_data[rsr_name]['spatial'].reshape(1, -1)
        ).flatten()
    
    for rsr_name in normalized_target_data:
        normalized_target_data[rsr_name]['spatial'] = spatial_scaler.transform(
            normalized_target_data[rsr_name]['spatial'].reshape(1, -1)
        ).flatten()
    
    # Prepare final processed data structure
    processed_data = {
        'source_domains': normalized_source_data,
        'target_domains': normalized_target_data,
        'source_scalers': source_scalers,
        'target_scalers': target_scalers,
        'spatial_scaler': spatial_scaler,
        'spatial_data': spatial_data,
        'params': {
            'input_features': _input_features,
            'days_x': _days_x,
            'days_y': _days_y,
            'time_range': ('1999-01-01', '2011-12-31'),
            'target_reservoirs': target_reservoirs,
            'source_reservoirs': list(normalized_source_data.keys())
        }
    }
    
    # Save processed data
    output_file = os.path.join(output_path, "dg_preprocessed_data.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nProcessed data saved to {output_file}")
    print(f"Source domains: {list(normalized_source_data.keys())}")
    print(f"Target domains: {list(normalized_target_data.keys())}")
    
    return processed_data



def prepare_dg_tensors(processed_data, domain_split='train'):
    """Prepare PyTorch tensors for training/evaluation
    Args:
        processed_data (dict): Output from preprocess_dg_data
        domain_split (str): 'train' or 'test'
    Returns:
        tuple: (X_tensor, y_tensor, spatial_tensor, reservoir_labels)
    """
    if domain_split == 'train':
        domains = processed_data['source_domains']
    elif domain_split == 'test':
        domains = processed_data['target_domains']
    else:
        raise ValueError(f"Invalid domain_split: {domain_split}. Use 'train' or 'test'.")
    
    X_list = []
    y_list = []
    spatial_list = []
    reservoir_labels = []
    
    for rsr_name, data in domains.items():
        n_samples = len(data['X'])
        X_list.append(torch.tensor(data['X'], dtype=torch.float32))
        y_list.append(torch.tensor(data['y'], dtype=torch.float32))
        # Repeat spatial vector for each sample
        spatial_repeated = np.tile(data['spatial'], (n_samples, 1))
        spatial_list.append(torch.tensor(spatial_repeated, dtype=torch.float32))
        reservoir_labels.extend([rsr_name] * n_samples)
    
    if len(X_list) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0), []
    
    X_tensor = torch.cat(X_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    spatial_tensor = torch.cat(spatial_list, dim=0)
    
    return X_tensor, y_tensor, spatial_tensor, reservoir_labels


if __name__ == "__main__":
    # Set up paths relative to project structure
    data_path = "./AdaTrip/data"
    output_path = "./HydroDCM/data"
    
    print("="*60)
    print("HydroDCM Domain Generalization Preprocessing")
    print("="*60)
    
    # Step 1: Preprocess data with reservoir-based domain splits
    processed_data = preprocess_dg_data(
        data_path=data_path,
        output_path=output_path,
        _input_features=3,  # temperature, precipitation, inflow
        _days_x=30,         # 30-day input window
        _days_y=7,          # 7-day prediction (multi-step ahead)
        target_reservoirs=['MCR', 'JVR', 'MCP']
    )
    
    # Step 2: Source domains are used for training, target domains for testing
    # No validation split needed as specified
    
    # Step 3: Save final processed data
    final_output_file = os.path.join(output_path, "dg_final_data.pkl")
    with open(final_output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Step 4: Prepare tensor data for training and testing
    print("\nPreparing tensor data for training and testing...")
    
    X_train, y_train, spatial_train, train_labels = prepare_dg_tensors(processed_data, 'train')
    X_test, y_test, spatial_test, test_labels = prepare_dg_tensors(processed_data, 'test')
    
    # Save tensor data
    tensor_data = {
        'X_train': X_train,
        'y_train': y_train,
        'spatial_train': spatial_train,
        'train_labels': train_labels,
        'X_test': X_test,
        'y_test': y_test,
        'spatial_test': spatial_test,
        'test_labels': test_labels
    }
    
    tensor_output_file = os.path.join(output_path, "dg_tensor_data.pt")
    torch.save(tensor_data, tensor_output_file)
    
    print(f"\nFinal summary:")
    print(f"Train data: {X_train.shape} samples from {len(set(train_labels))} reservoirs")
    print(f"Train spatial: {spatial_train.shape}")
    print(f"Test data: {X_test.shape} samples from {len(set(test_labels))} reservoirs")
    print(f"Test spatial: {spatial_test.shape}")
    print(f"\nFiles saved:")
    print(f"  - {final_output_file}")
    print(f"  - {tensor_output_file}")
    print("="*60)