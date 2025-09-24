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
import gc
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

import sys
sys.path.append(r"D:\NCKH.2025-2026\LiquidPredictLiquid")
from LiquidMamba import LiquidMambaAttention, MultiScaleLiquidMamba
from unet_model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Memory management utilities
def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0

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

# Optimized image loading with smaller target size
def load_mask_images(mask_path, target_size=(224, 224)):  # Reduced from 320x320
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
interpolated_images = interpolated_images.reshape(-1, 1, 224, 224)  # Reduced size
print(f"Interpolated images shape: {interpolated_images.shape}")

# Load and preprocess time series data
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

# Memory-optimized GeM pooling
class GeM(nn.Module):
    """Generalized Mean Pooling: mặc định m=3.0; m=1 -> GAP, m->inf -> GMP"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.mean(x, dim=(-2, -1)).pow(1.0 / self.p)

# Memory-optimized UNet feature extractor
class LightweightUNetMSFeatureExtractor(nn.Module):
    """
    Lightweight UNet-based Multi-Scale Feature Extractor
    Optimized for memory efficiency while maintaining performance
    """
    def __init__(self, input_channels=1, output_features=512, freeze_backbone=True):  # Reduced output_features
        super().__init__()
        
        # Initialize lighter UNet model
        self.unet = UNet(
            n_channels=input_channels, 
            n_classes=2,
            bilinear=True,  # Use bilinear upsampling to save memory
            c3str_layers=1,  # Reduced from 2
            c3str_heads=4,   # Reduced from 8
            c3str_mlp_ratio=2,  # Reduced from 4
            c3str_dropout=0.1
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.unet, 'gradient_checkpointing_enable'):
            self.unet.gradient_checkpointing_enable()
        
        self.pool = GeM(p=3.0)
        
        # Calculate feature dimensions with smaller model
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)  # Reduced size
            x1 = self.unet.inc(dummy_input)
            x2 = self.unet.down1(x1)
            x3 = self.unet.down2(x2)
            x4 = self.unet.down3(x3)
            x5 = self.unet.down4(x4)
            
            c3, c4, c5 = x3.shape[1], x4.shape[1], x5.shape[1]
            concat_dim = c3 + c4 + c5
        
        print(f"Lightweight UNet feature dimensions - x3: {c3}, x4: {c4}, x5: {c5}, total: {concat_dim}")
        
        self.norm = nn.LayerNorm(concat_dim)
        # Simplified head to reduce parameters
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, output_features),
            nn.ReLU(inplace=True),
        )
        
        if freeze_backbone:
            for p in self.unet.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True
        
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # Use gradient checkpointing during training
        if self.training:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward_inference(x)
    
    def _forward_inference(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)
        
        z3 = self.pool(x3)
        z4 = self.pool(x4)
        z5 = self.pool(x5)
        
        z = torch.cat([z3, z4, z5], dim=1)
        z = self.norm(z)
        z = self.head(z)
        return z
    
    def _forward_with_checkpointing(self, x):
        # Manual gradient checkpointing for memory efficiency
        x1 = torch.utils.checkpoint.checkpoint(self.unet.inc, x)
        x2 = torch.utils.checkpoint.checkpoint(self.unet.down1, x1)
        x3 = torch.utils.checkpoint.checkpoint(self.unet.down2, x2)
        x4 = torch.utils.checkpoint.checkpoint(self.unet.down3, x3)
        x5 = torch.utils.checkpoint.checkpoint(self.unet.down4, x4)
        
        z3 = self.pool(x3)
        z4 = self.pool(x4)
        z5 = self.pool(x5)
        
        z = torch.cat([z3, z4, z5], dim=1)
        z = self.norm(z)
        z = self.head(z)
        return z

# Initialize memory-optimized model
clear_memory()
unet_model = LightweightUNetMSFeatureExtractor(input_channels=1, output_features=512, freeze_backbone=True).to(device)
print("Lightweight UNet model built")
get_memory_usage()

# Memory-optimized feature extraction
def extract_image_features(model, images, batch_size=2):  # Reduced batch size
    model.eval()
    features = []
    
    clear_memory()  # Clear memory before processing
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            batch_features = model(batch_tensor)
            features.append(batch_features.cpu().numpy())
            
            # Clear intermediate tensors
            del batch_tensor, batch_features
            if i % (batch_size * 4) == 0:  # Clear cache periodically
                clear_memory()
    
    return np.vstack(features)

image_features = extract_image_features(unet_model, interpolated_images)
print(f"Image features shape: {image_features.shape}")
clear_memory()

# Lightweight time series feature extractor
class LightweightTimeSeriesFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_features=512):  # Reduced output
        super(LightweightTimeSeriesFeatureExtractor, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 64),   # Reduced from 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 128),         # Reduced from 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 256),        # Reduced from 512
            nn.ReLU(inplace=True),
            nn.Linear(256, output_features),
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

ts_model = LightweightTimeSeriesFeatureExtractor(len(feature_columns)).to(device)

def extract_ts_features(model, data, batch_size=16):  # Reduced batch size
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_features = model(batch_tensor)
            features.append(batch_features.cpu().numpy())
            
            del batch_tensor, batch_features
            if i % (batch_size * 8) == 0:
                clear_memory()
    
    return np.vstack(features)

ts_features = extract_ts_features(ts_model, ts_feature_data_scaled)
print(f"Time series features shape: {ts_features.shape}")

# Memory-optimized feature expansion and combination
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

# Process features in smaller chunks to avoid memory issues
feature_scaler = StandardScaler()
image_features_norm = feature_scaler.fit_transform(expanded_image_features)
ts_features_norm = feature_scaler.fit_transform(ts_features)

combined_features = np.concatenate([image_features_norm, ts_features_norm], axis=1)
print(f"Combined features shape: {combined_features.shape}")
clear_memory()

# Create sequences with shorter time steps
def create_sequences_for_liquidmamba(features, labels, time_steps=3):  # Reduced from 4
    X, y = [], []
    
    for i in range(time_steps, len(features)):
        X.append(features[i-time_steps:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)

labels = df['WaterLevel_m'].values
min_len = min(len(combined_features), len(labels))
combined_features = combined_features[:min_len]
labels = labels[:min_len]

X_seq, y_seq = create_sequences_for_liquidmamba(combined_features, labels, time_steps=3)
print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")

# Split data
train_size = int(0.7 * len(X_seq))
val_size = int(0.2 * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

baseline_ts_sequences, _ = create_sequences_for_liquidmamba(ts_features, labels, time_steps=3)
baseline_X_train = baseline_ts_sequences[:train_size]
baseline_X_val = baseline_ts_sequences[train_size:train_size+val_size]
baseline_X_test = baseline_ts_sequences[train_size+val_size:]

# Memory-optimized dataset
class MemoryOptimizedWaterLevelDataset(Dataset):
    def __init__(self, sequences, targets, scaler_y=None, fit_scaler=False):
        # Convert to half precision to save memory
        self.sequences = torch.FloatTensor(sequences).half()
        
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
            
        self.targets = torch.FloatTensor(targets_scaled).half()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx].float(), self.targets[idx].float()

scaler_y = MinMaxScaler()
train_dataset = MemoryOptimizedWaterLevelDataset(X_train, y_train, scaler_y, fit_scaler=True)
val_dataset = MemoryOptimizedWaterLevelDataset(X_val, y_val, scaler_y)
test_dataset = MemoryOptimizedWaterLevelDataset(X_test, y_test, scaler_y)

baseline_train_dataset = MemoryOptimizedWaterLevelDataset(baseline_X_train, y_train, scaler_y)
baseline_val_dataset = MemoryOptimizedWaterLevelDataset(baseline_X_val, y_val, scaler_y)
baseline_test_dataset = MemoryOptimizedWaterLevelDataset(baseline_X_test, y_test, scaler_y)

# Smaller batch sizes
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)  # Reduced
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

baseline_train_loader = DataLoader(baseline_train_dataset, batch_size=8, shuffle=True, pin_memory=True)
baseline_val_loader = DataLoader(baseline_val_dataset, batch_size=8, shuffle=False, pin_memory=True)
baseline_test_loader = DataLoader(baseline_test_dataset, batch_size=8, shuffle=False, pin_memory=True)

print("Memory-optimized data loaders created")
clear_memory()
get_memory_usage()

# Memory-optimized LiquidMamba models
class MemoryOptimizedLiquidMambaModel(nn.Module):
    """
    Memory-optimized LiquidMamba model with gradient checkpointing and reduced parameters
    """
    def __init__(self, feature_dim, d_model=64, num_scales=2, state_size=8):  # Reduced parameters
        super(MemoryOptimizedLiquidMambaModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Lightweight Multi-scale Liquid Mamba
        self.liquid_mamba = MultiScaleLiquidMamba(
            d_model=d_model,
            num_scales=num_scales,  # Reduced from 3
            state_size=state_size,  # Reduced from 16
            expand_factor=1         # Reduced from 2
        )
        
        # Simplified output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),  # Reduced from 64
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Use mixed precision and gradient checkpointing during training
        if self.training:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward_inference(x)
    
    def _forward_inference(self, x):
        x = self.input_projection(x)
        mamba_out, _ = self.liquid_mamba(x)
        last_output = mamba_out[:, -1, :]
        output = self.layer_norm(last_output)
        output = self.dropout(output)
        output = self.fc(output)
        return output
    
    def _forward_with_checkpointing(self, x):
        # Apply gradient checkpointing for memory efficiency
        x = torch.utils.checkpoint.checkpoint(self.input_projection, x)
        mamba_out, _ = torch.utils.checkpoint.checkpoint(self.liquid_mamba, x)
        last_output = mamba_out[:, -1, :]
        output = self.layer_norm(last_output)
        output = self.dropout(output)
        output = self.fc(output)
        return output
    
    def reset_liquid_states(self):
        """Reset all liquid time constant states"""
        if hasattr(self.liquid_mamba, 'reset_liquid_state'):
            self.liquid_mamba.reset_liquid_state()

class MemoryOptimizedBaselineModel(nn.Module):
    """
    Memory-optimized baseline model
    """
    def __init__(self, feature_dim, d_model=32, state_size=4):  # Even smaller
        super(MemoryOptimizedBaselineModel, self).__init__()
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        self.liquid_mamba = LiquidMambaAttention(
            d_model=d_model,
            state_size=state_size,
            expand_factor=1,
            use_liquid_scaling=True
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        mamba_out, _ = self.liquid_mamba(x)
        last_output = mamba_out[:, -1, :]
        output = self.layer_norm(last_output)
        output = self.dropout(output)
        output = self.fc(output)
        return output
    
    def reset_liquid_states(self):
        """Reset liquid time constant states"""
        if hasattr(self.liquid_mamba, 'reset_liquid_state'):
            self.liquid_mamba.reset_liquid_state()

# Initialize models with memory monitoring
clear_memory()
print("Creating memory-optimized models...")

proposed_model = MemoryOptimizedLiquidMambaModel(X_seq.shape[2]).to(device)
baseline_model = MemoryOptimizedBaselineModel(baseline_ts_sequences.shape[2]).to(device)

print("Memory-optimized LiquidMamba models built")
print(f"Proposed model input shape: {X_seq.shape}")
print(f"Baseline model input shape: {baseline_ts_sequences.shape}")
print(f"Proposed model parameters: {sum(p.numel() for p in proposed_model.parameters()):,}")
print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

get_memory_usage()

# Memory-optimized training function with mixed precision
def memory_optimized_train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, reset_frequency=10):
    """
    Memory-optimized training function with mixed precision and gradient accumulation
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()  # For mixed precision training
    
    train_losses = []
    val_losses = []
    
    # Gradient accumulation for effective larger batch size
    accumulation_steps = 4
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Reset liquid states periodically
        if epoch % reset_frequency == 0 and hasattr(model, 'reset_liquid_states'):
            model.reset_liquid_states()
        
        # Clear memory at start of epoch
        clear_memory()
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, targets) / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                del sequences, targets, outputs
                clear_memory()
        
        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Validation with memory optimization
        model.eval()
        val_loss = 0.0
        
        if hasattr(model, 'reset_liquid_states'):
            model.reset_liquid_states()
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(sequences).squeeze()
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                del sequences, targets, outputs
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            get_memory_usage()
        
        # Clear memory at end of epoch
        clear_memory()
    
    return train_losses, val_losses

# Memory-optimized prediction function
def memory_optimized_make_predictions(model, test_loader, scaler_y):
    """
    Memory-optimized prediction with automatic memory cleanup
    """
    model.eval()
    predictions = []
    actuals = []
    
    if hasattr(model, 'reset_liquid_states'):
        model.reset_liquid_states()
    
    clear_memory()
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(sequences).squeeze()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
            
            del sequences, targets, outputs
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions_unscaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_unscaled = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    clear_memory()
    return predictions_unscaled, actuals_unscaled

print("\nStarting memory-optimized training...")
print("="*50)

# Train proposed model
print("Training proposed memory-optimized LiquidMamba model...")
get_memory_usage()
proposed_train_losses, proposed_val_losses = memory_optimized_train_model(
    proposed_model, train_loader, val_loader, num_epochs=20  # Reduced epochs for testing
)

clear_memory()
print("\nTraining baseline memory-optimized LiquidMamba model...")
get_memory_usage()
baseline_train_losses, baseline_val_losses = memory_optimized_train_model(
    baseline_model, baseline_train_loader, baseline_val_loader, num_epochs=20
)

clear_memory()

# Make predictions
print("\nMaking predictions...")
y_pred_proposed, y_test_actual = memory_optimized_make_predictions(proposed_model, test_loader, scaler_y)
y_pred_baseline, _ = memory_optimized_make_predictions(baseline_model, baseline_test_loader, scaler_y)

print("Memory-optimized LiquidMamba predictions completed")
print(f"Predicted shapes - Proposed: {y_pred_proposed.shape}, Baseline: {y_pred_baseline.shape}")
print(f"Actual test shape: {y_test_actual.shape}")

clear_memory()
get_memory_usage()

# Calculate metrics
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

print("\n" + "="*60)
print("MEMORY-OPTIMIZED MODEL RESULTS")
print("="*60)

print("BASELINE LIQUIDMAMBA MODEL RESULTS:")
print(f"MAE:  {baseline_metrics['MAE']:.4f}")
print(f"MSE:  {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")

print(f"\nPROPOSED MODEL (Lightweight UNet + MultiScale LiquidMamba) RESULTS:")
print(f"MAE:  {proposed_metrics['MAE']:.4f}")
print(f"MSE:  {proposed_metrics['MSE']:.4f}")
print(f"RMSE: {proposed_metrics['RMSE']:.4f}")

print(f"\nIMPROVEMENT OVER BASELINE:")
print(f"MAE improvement:  {mae_improvement:.1f}%")
print(f"MSE improvement:  {mse_improvement:.1f}%")
print(f"RMSE improvement: {rmse_improvement:.1f}%")

print(f"\nFinal memory usage:")
get_memory_usage()

# Save models
print("\nSaving memory-optimized models...")
torch.save(proposed_model.state_dict(), 'memory_optimized_unet_liquidmamba_model.pth')
torch.save(baseline_model.state_dict(), 'memory_optimized_baseline_liquidmamba_model.pth')
print("Memory-optimized UNet + LiquidMamba models saved successfully!")

clear_memory()
print("Training completed successfully with memory optimization!")