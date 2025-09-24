import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(r"D:\NCKH.2025-2026\LiquidPredictLiquid")
from LiquidMamba import (
    LiquidMambaAttention, 
    MultiScaleLiquidMamba)

# Import UNet model
from unet_model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

image_original_path = r"D:\NCKH.2025-2026\profNgan\Image_AnKhe_Goc (1)\image_original"
image_mask_path = r"D:\NCKH.2025-2026\profNgan\Image_AnKhe_Goc (1)\images_mask_AnKhe"
label_path = r"D:\NCKH.2025-2026\profNgan\Image_AnKhe_Goc (1)\label_AnKhe_goc"
csv_path = r"D:\NCKH.2025-2026\profNgan\DataSet\data_so_AnKhe\AnKhe.csv"

print(f"Image original path exists: {os.path.exists(image_original_path)}")
print(f"Image mask path exists: {os.path.exists(image_mask_path)}")
print(f"Label path exists: {os.path.exists(label_path)}")
print(f"CSV path exists: {os.path.exists(csv_path)}")

# Cell 3: Load and preprocess images
def load_mask_images(mask_path, target_size=(320, 320)):
    image_files = glob.glob(os.path.join(mask_path, "*"))
    images = []
    filenames = []
    
    for img_path in sorted(image_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            images.append(img_normalized)
            filenames.append(os.path.basename(img_path))
    
    return np.array(images), filenames

mask_images, image_filenames = load_mask_images(image_mask_path)
print(f"Loaded {len(mask_images)} mask images")
print(f"Image shape: {mask_images.shape}")

# Cell 4: Interpolate images to monthly coverage
def interpolate_images_to_monthly(images, target_months=48):
    if len(images) >= target_months:
        return images[:target_months]
    
    interpolated = []
    images_per_month = len(images) / 12.0
    
    for month in range(target_months):
        year = month // 12
        month_in_year = month % 12
        
        base_idx = int(month_in_year * images_per_month) % len(images)
        base_image = images[base_idx].copy()
        
        seasonal_noise = 0.05 * np.sin(2 * np.pi * month_in_year / 12)
        noise = np.random.normal(0, 0.02, base_image.shape)
        
        interpolated_image = np.clip(base_image + seasonal_noise + noise, 0, 1)
        interpolated.append(interpolated_image)
    
    return np.array(interpolated)

interpolated_images = interpolate_images_to_monthly(mask_images, 48)
interpolated_images = interpolated_images.reshape(-1, 1, 320, 320)
print(f"Interpolated images shape: {interpolated_images.shape}")

# Cell 5: Load and preprocess time series data
df = pd.read_csv(csv_path)
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

df.columns = ['Time', 'WaterLevel_m', 'TotalDischarge_m3s', 'Inflow_m3s']

df['Month'] = df['Time'].dt.month
df['IsFloodSeason'] = ((df['Month'] >= 5) & (df['Month'] <= 10)).astype(int)

for lag in [1, 2, 3]:
    df[f'WaterLevel_lag{lag}'] = df['WaterLevel_m'].shift(lag)
    df[f'Inflow_lag{lag}'] = df['Inflow_m3s'].shift(lag)

df = df.dropna().reset_index(drop=True)
print(f"Time series data shape: {df.shape}")
print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

# Cell 6: Define VGG19 feature extractor with GeM and Multi-Scale features
class GeM(nn.Module):
    """Generalized Mean Pooling: mặc định m=3.0; m=1 -> GAP, m->inf -> GMP"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.mean(x, dim=(-2, -1)).pow(1.0 / self.p)  # (B,C)

class VGGMSFeatureExtractor(nn.Module):
    """
    VGG-based Multi-Scale Feature Extractor
    Rút đặc trưng từ các tầng VGG: conv3_x, conv4_x, conv5_x (multi-scale) -> concat -> head -> 1024-d.
    - Đóng băng backbone để ổn định khi chỉ train GRU.
    - Dùng GeM (ổn định hơn GAP) + LayerNorm + MLP nhỏ.
    """
    def __init__(self, input_channels=1, output_features=1024, freeze_backbone=True):
        super().__init__()
        
        # VGG19-like backbone
        # Block 1: 64 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 320->160
        )
        
        # Block 2: 128 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 160->80
        )
        
        # Block 3: 256 channels (multi-scale feature 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 80->40
        )
        
        # Block 4: 512 channels (multi-scale feature 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 40->20
        )
        
        # Block 5: 512 channels (multi-scale feature 3)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 20->10
        )
        
        # GeM pooling for each scale
        self.pool = GeM(p=3.0)
        
        # Calculate concatenated dimension: 256 + 512 + 512 = 1280
        concat_dim = 256 + 512 + 512
        
        self.norm = nn.LayerNorm(concat_dim)
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 1024), 
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_features), 
            nn.ReLU(inplace=True),
        )
        
        if freeze_backbone:
            for p in self.parameters():
                if p is not self.norm.weight and p is not self.norm.bias:
                    p.requires_grad = False
            # Chỉ cho phép train head và norm
            for p in self.head.parameters():
                p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True
        
        self.eval()  # mặc định đặt eval để BN/Dropout cố định khi trích xuất

    @torch.no_grad()
    def forward(self, x):
        # Forward through VGG blocks
        x = self.block1(x)  # 64 channels
        x = self.block2(x)  # 128 channels
        x3 = self.block3(x)  # 256 channels, 40x40
        x4 = self.block4(x3)  # 512 channels, 20x20  
        x5 = self.block5(x4)  # 512 channels, 10x10
        
        # GeM pooling on multi-scale features
        z3 = self.pool(x3)  # (B, 256)
        z4 = self.pool(x4)  # (B, 512)
        z5 = self.pool(x5)  # (B, 512)
        
        # Concatenate multi-scale features
        z = torch.cat([z3, z4, z5], dim=1)  # (B, 1280)
        z = self.norm(z)
        z = self.head(z)  # (B, 1024)
        return z

# khởi tạo VGG extractor (đóng băng backbone, chỉ rút đặc trưng)  
vgg19_model = VGGMSFeatureExtractor(input_channels=1, output_features=1024, freeze_backbone=True).to(device)
print("VGG19 model built")

# Cell 7: Extract image features
def extract_image_features(model, images, batch_size=4):
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_features = model(batch_tensor)
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

image_features = extract_image_features(vgg19_model, interpolated_images)
print(f"Image features shape: {image_features.shape}")

# Cell 8: Define time series feature extractor
class TimeSeriesFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_features=1024):
        super(TimeSeriesFeatureExtractor, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.fc_layers(x)

feature_columns = ['Inflow_m3s', 'TotalDischarge_m3s', 'IsFloodSeason', 
                  'WaterLevel_lag1', 'Inflow_lag1', 'WaterLevel_lag2', 
                  'Inflow_lag2', 'WaterLevel_lag3', 'Inflow_lag3']

ts_feature_data = df[feature_columns].values
ts_scaler = StandardScaler()
ts_feature_data_scaled = ts_scaler.fit_transform(ts_feature_data)

ts_model = TimeSeriesFeatureExtractor(len(feature_columns)).to(device)

def extract_ts_features(model, data, batch_size=32):
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_features = model(batch_tensor)
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

ts_features = extract_ts_features(ts_model, ts_feature_data_scaled)
print(f"Time series features shape: {ts_features.shape}")

# Cell 9: Expand image features to daily resolution
def expand_image_features_to_daily(image_features, n_days):
    n_months = len(image_features)
    days_per_month = n_days / n_months
    
    expanded_features = []
    for i, monthly_feature in enumerate(image_features):
        start_day = int(i * days_per_month)
        end_day = int((i + 1) * days_per_month)
        
        for _ in range(end_day - start_day):
            expanded_features.append(monthly_feature)
    
    return np.array(expanded_features[:n_days])

expanded_image_features = expand_image_features_to_daily(image_features, len(ts_features))
print(f"Expanded image features shape: {expanded_image_features.shape}")

# Cell 10: Combine multimodal features
feature_scaler = StandardScaler()
image_features_norm = feature_scaler.fit_transform(expanded_image_features)
ts_features_norm = feature_scaler.fit_transform(ts_features)

combined_features = np.concatenate([image_features_norm, ts_features_norm], axis=1)
print(f"Combined features shape: {combined_features.shape}")

# Cell 11: Create sequences for GRU
def create_sequences_for_gru(features, labels, time_steps=4):
    X, y = [], []
    
    for i in range(time_steps, len(features)):
        X.append(features[i-time_steps:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)

labels = df['WaterLevel_m'].values
min_len = min(len(combined_features), len(labels))
combined_features = combined_features[:min_len]
labels = labels[:min_len]

X_seq, y_seq = create_sequences_for_gru(combined_features, labels, time_steps=4)
print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")

# Cell 12: Split data
train_size = int(0.7 * len(X_seq))
val_size = int(0.2 * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

baseline_ts_sequences, _ = create_sequences_for_gru(ts_features, labels, time_steps=4)
baseline_X_train = baseline_ts_sequences[:train_size]
baseline_X_val = baseline_ts_sequences[train_size:train_size+val_size]
baseline_X_test = baseline_ts_sequences[train_size+val_size:]

# Cell 13: Define Dataset class
class WaterLevelDataset(Dataset):
    def __init__(self, sequences, targets, scaler_y=None, fit_scaler=False):
        self.sequences = torch.FloatTensor(sequences)
        
        if fit_scaler:
            if scaler_y is None:
                self.scaler_y = MinMaxScaler()
            else:
                self.scaler_y = scaler_y
            targets_scaled = self.scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()
        elif scaler_y is not None:
            self.scaler_y = scaler_y
            targets_scaled = scaler_y.transform(targets.reshape(-1, 1)).flatten()
        else:
            self.scaler_y = None
            targets_scaled = targets
            
        self.targets = torch.FloatTensor(targets_scaled)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

scaler_y = MinMaxScaler()
train_dataset = WaterLevelDataset(X_train, y_train, scaler_y, fit_scaler=True)
val_dataset = WaterLevelDataset(X_val, y_val, scaler_y)
test_dataset = WaterLevelDataset(X_test, y_test, scaler_y)

baseline_train_dataset = WaterLevelDataset(baseline_X_train, y_train, scaler_y)
baseline_val_dataset = WaterLevelDataset(baseline_X_val, y_val, scaler_y)
baseline_test_dataset = WaterLevelDataset(baseline_X_test, y_test, scaler_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

baseline_train_loader = DataLoader(baseline_train_dataset, batch_size=32, shuffle=True)
baseline_val_loader = DataLoader(baseline_val_dataset, batch_size=32, shuffle=False)
baseline_test_loader = DataLoader(baseline_test_dataset, batch_size=32, shuffle=False)

# Cell 14: Define LiquidMamba models
class ProposedLiquidMambaModel(nn.Module):
    """
    Proposed model using MultiScaleLiquidMamba for advanced temporal modeling
    """
    def __init__(self, feature_dim, d_model=128, num_scales=3, state_size=16):
        super(ProposedLiquidMambaModel, self).__init__()
        
        # Input projection to d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Multi-scale Liquid Mamba attention
        self.liquid_mamba = MultiScaleLiquidMamba(
            d_model=d_model,
            num_scales=num_scales,
            state_size=state_size,
            expand_factor=2
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Apply Multi-scale Liquid Mamba
        mamba_out, _ = self.liquid_mamba(x)  # (batch, seq, d_model)
        
        # Use last output for prediction (similar to GRU approach)
        last_output = mamba_out[:, -1, :]  # (batch, d_model)
        
        # Apply normalization and dropout
        output = self.layer_norm(last_output)
        output = self.dropout(output)
        
        # Final prediction
        output = self.fc(output)
        return output
    
    def reset_liquid_states(self):
        """Reset all liquid time constant states"""
        self.liquid_mamba.reset_liquid_state()

class BaselineLiquidMambaModel(nn.Module):
    def __init__(self, feature_dim, d_model=64, state_size=8):
        super(BaselineLiquidMambaModel, self).__init__()
        
        # Input projection to d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Single-scale Liquid Mamba attention
        self.liquid_mamba = LiquidMambaAttention(
            d_model=d_model,
            state_size=state_size,
            expand_factor=1,  # Smaller expand factor for baseline
            use_liquid_scaling=True
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Apply Liquid Mamba
        mamba_out, _ = self.liquid_mamba(x)  # (batch, seq, d_model)
        
        # Use last output for prediction
        last_output = mamba_out[:, -1, :]  # (batch, d_model)
        
        # Apply normalization and dropout
        output = self.layer_norm(last_output)
        output = self.dropout(output)
        
        # Final prediction
        output = self.fc(output)
        return output
    
    def reset_liquid_states(self):
        """Reset liquid time constant states"""
        self.liquid_mamba.reset_liquid_state()

proposed_model = ProposedLiquidMambaModel(X_seq.shape[2]).to(device)
baseline_model = BaselineLiquidMambaModel(baseline_ts_sequences.shape[2]).to(device)

print("LiquidMamba Models built")
print(f"Proposed LiquidMamba model input shape: {X_seq.shape}")
print(f"Baseline LiquidMamba model input shape: {baseline_ts_sequences.shape}")
print(f"Proposed model parameters: {sum(p.numel() for p in proposed_model.parameters()):,}")
print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

# Cell 15: Enhanced training function for LiquidMamba models
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, reset_frequency=10):
    """
    Enhanced training function for LiquidMamba models with liquid state management
    
    Args:
        model: LiquidMamba-based model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        reset_frequency: How often to reset liquid states (in epochs)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Reset liquid states periodically for training stability
        if epoch % reset_frequency == 0 and hasattr(model, 'reset_liquid_states'):
            model.reset_liquid_states()
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability with Mamba
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        
        # Reset liquid states before validation for consistent evaluation
        if hasattr(model, 'reset_liquid_states'):
            model.reset_liquid_states()
            
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

print("Training proposed LiquidMamba model...")
proposed_train_losses, proposed_val_losses = train_model(proposed_model, train_loader, val_loader)

print("Training baseline LiquidMamba model...")
baseline_train_losses, baseline_val_losses = train_model(baseline_model, baseline_train_loader, baseline_val_loader)

# Cell 16: Enhanced prediction function for LiquidMamba models
def make_predictions(model, test_loader, scaler_y):
    """
    Enhanced prediction function for LiquidMamba models with state management
    """
    model.eval()
    predictions = []
    actuals = []
    
    # Reset liquid states before making predictions for consistency
    if hasattr(model, 'reset_liquid_states'):
        model.reset_liquid_states()
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions_unscaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_unscaled = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    return predictions_unscaled, actuals_unscaled

y_pred_proposed, y_test_actual = make_predictions(proposed_model, test_loader, scaler_y)
y_pred_baseline, _ = make_predictions(baseline_model, baseline_test_loader, scaler_y)

print("LiquidMamba predictions completed")
print(f"Predicted shapes - Proposed: {y_pred_proposed.shape}, Baseline: {y_pred_baseline.shape}")
print(f"Actual test shape: {y_test_actual.shape}")

# Cell 17: Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

proposed_metrics = calculate_metrics(y_test_actual, y_pred_proposed)
baseline_metrics = calculate_metrics(y_test_actual, y_pred_baseline)

mae_improvement = (baseline_metrics['MAE'] - proposed_metrics['MAE']) / baseline_metrics['MAE'] * 100
mse_improvement = (baseline_metrics['MSE'] - proposed_metrics['MSE']) / baseline_metrics['MSE'] * 100
rmse_improvement = (baseline_metrics['RMSE'] - proposed_metrics['RMSE']) / baseline_metrics['RMSE'] * 100

print("BASELINE LIQUIDMAMBA MODEL RESULTS:")
print(f"MAE:  {baseline_metrics['MAE']:.4f}")
print(f"MSE:  {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")

print(f"\nPROPOSED MODEL (VGG19 + MultiScale LiquidMamba) RESULTS:")
print(f"MAE:  {proposed_metrics['MAE']:.4f}")
print(f"MSE:  {proposed_metrics['MSE']:.4f}")
print(f"RMSE: {proposed_metrics['RMSE']:.4f}")

print(f"\nIMPROVEMENT OVER BASELINE:")
print(f"MAE improvement:  {mae_improvement:.1f}%")
print(f"MSE improvement:  {mse_improvement:.1f}%")
print(f"RMSE improvement: {rmse_improvement:.1f}%")

# Cell 18: Visualize results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(y_test_actual, label='Actual', alpha=0.8)
plt.plot(y_pred_proposed, label='MultiScale LiquidMamba', alpha=0.8)
plt.title('Proposed MultiScale LiquidMamba Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(y_test_actual, label='Actual', alpha=0.8)
plt.plot(y_pred_baseline, label='Baseline LiquidMamba', alpha=0.8)
plt.title('Baseline LiquidMamba Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
models = ['Baseline LiquidMamba', 'MultiScale LiquidMamba']
mae_values = [baseline_metrics['MAE'], proposed_metrics['MAE']]
mse_values = [baseline_metrics['MSE'], proposed_metrics['MSE']]
rmse_values = [baseline_metrics['RMSE'], proposed_metrics['RMSE']]

x = np.arange(len(models))
width = 0.25

plt.bar(x - width, mae_values, width, label='MAE', alpha=0.8)
plt.bar(x, mse_values, width, label='MSE', alpha=0.8)
plt.bar(x + width, rmse_values, width, label='RMSE', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Error Values')
plt.title('LiquidMamba Model Comparison')
plt.xticks(x, models, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.scatter(y_test_actual, y_pred_proposed, alpha=0.6, label='MultiScale LiquidMamba')
plt.scatter(y_test_actual, y_pred_baseline, alpha=0.6, label='Baseline LiquidMamba')
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Water Level (m)')
plt.ylabel('Predicted Water Level (m)')
plt.title('Actual vs Predicted (LiquidMamba)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 19: Training history visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(proposed_train_losses, label='MultiScale LiquidMamba Training Loss')
plt.plot(proposed_val_losses, label='MultiScale LiquidMamba Validation Loss')
plt.title('Proposed MultiScale LiquidMamba Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(baseline_train_losses, label='Baseline LiquidMamba Training Loss')
plt.plot(baseline_val_losses, label='Baseline LiquidMamba Validation Loss')
plt.title('Baseline LiquidMamba Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(df['Time'][:1000], df['WaterLevel_m'][:1000])
plt.title('Water Level Time Series (First 1000 points)')
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
monthly_avg = df.groupby(df['Month'])['WaterLevel_m'].mean()
plt.bar(monthly_avg.index, monthly_avg.values)
plt.title('Average Water Level by Month')
plt.xlabel('Month')
plt.ylabel('Average Water Level (m)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 20: Additional analysis and save results
flood_data = df[df['IsFloodSeason'] == 1]['WaterLevel_m']
dry_data = df[df['IsFloodSeason'] == 0]['WaterLevel_m']

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.hist(flood_data, alpha=0.7, label='Flood Season', bins=30)
plt.hist(dry_data, alpha=0.7, label='Dry Season', bins=30)
plt.title('Water Level Distribution by Season')
plt.xlabel('Water Level (m)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sample_images = interpolated_images[:4].reshape(4, 320, 320)
for i in range(4):
    plt.subplot(2, 4, 4 + i + 1)
    plt.imshow(sample_images[i], cmap='gray')
    plt.title(f'Mask {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

results = {
    'proposed_metrics': proposed_metrics,
    'baseline_metrics': baseline_metrics,
    'improvements': {
        'MAE': mae_improvement,
        'MSE': mse_improvement,
        'RMSE': rmse_improvement
    },
    'predictions': {
        'actual': y_test_actual,
        'proposed': y_pred_proposed,
        'baseline': y_pred_baseline
    }
}

print("LiquidMamba experiment completed successfully!")
print(f"Final LiquidMamba results: MAE {mae_improvement:.1f}%, MSE {mse_improvement:.1f}%, RMSE {rmse_improvement:.1f}% improvement")

torch.save(proposed_model.state_dict(), 'proposed_vgg19_liquidmamba_model.pth')
torch.save(baseline_model.state_dict(), 'baseline_liquidmamba_model.pth')
print("LiquidMamba models saved successfully!")