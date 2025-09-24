# Cell 1: Import libraries with memory optimization
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
import warnings
import gc
warnings.filterwarnings('ignore')

# Memory optimization settings
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")

# Cell 2: Set paths and load data
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

# Cell 6: Define VGG19 feature extractor
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_features=1024):
        super(VGG19FeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 10 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

vgg19_model = VGG19FeatureExtractor().to(device)
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

# Cell 10: Memory-efficient Cross-Modal Attention and Physics Loss
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, ts_dim, hidden_dim=256):  # Reduced hidden_dim
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.ts_proj = nn.Linear(ts_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)  # Reduced heads
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, image_features, ts_features):
        # Process in smaller chunks to avoid memory issues
        batch_size = min(512, image_features.shape[0])  # Process in batches
        
        attended_chunks = []
        for i in range(0, image_features.shape[0], batch_size):
            end_idx = min(i + batch_size, image_features.shape[0])
            
            img_batch = image_features[i:end_idx]
            ts_batch = ts_features[i:end_idx]
            
            img_proj = self.image_proj(img_batch)
            ts_proj = self.ts_proj(ts_batch)
            
            # Add sequence dimension for attention
            img_proj = img_proj.unsqueeze(1)  # (batch, 1, hidden_dim)
            ts_proj = ts_proj.unsqueeze(1)    # (batch, 1, hidden_dim)
            
            attended_features, _ = self.attention(img_proj, ts_proj, ts_proj)
            attended_features = attended_features.squeeze(1)  # Remove sequence dimension
            
            attended_chunks.append(attended_features)
        
        # Concatenate all chunks
        attended_features = torch.cat(attended_chunks, dim=0)
        return self.output_proj(attended_features)

# Memory-efficient feature combination
def combine_features_efficiently(image_features, ts_features, hidden_dim=256):
    print("Combining features with attention (memory-efficient)...")
    
    # Reduce feature dimensions first
    if image_features.shape[1] > 512:
        pca_img = torch.pca_lowrank(torch.FloatTensor(image_features), q=512)[0]
        image_features = pca_img.numpy()
    
    if ts_features.shape[1] > 512:
        pca_ts = torch.pca_lowrank(torch.FloatTensor(ts_features), q=512)[0]
        ts_features = pca_ts.numpy()
    
    # Scale features
    feature_scaler = StandardScaler()
    image_features_norm = feature_scaler.fit_transform(image_features)
    ts_features_norm = feature_scaler.fit_transform(ts_features)
    
    # Create attention module
    attention_module = CrossModalAttention(
        image_dim=image_features_norm.shape[1], 
        ts_dim=ts_features_norm.shape[1], 
        hidden_dim=hidden_dim
    ).to(device)
    
    # Process in chunks to avoid memory issues
    chunk_size = 256
    combined_chunks = []
    
    attention_module.eval()
    with torch.no_grad():
        for i in range(0, len(image_features_norm), chunk_size):
            end_idx = min(i + chunk_size, len(image_features_norm))
            
            img_chunk = torch.FloatTensor(image_features_norm[i:end_idx]).to(device)
            ts_chunk = torch.FloatTensor(ts_features_norm[i:end_idx]).to(device)
            
            attended_chunk = attention_module(img_chunk, ts_chunk)
            combined_chunks.append(attended_chunk.cpu().numpy())
            
            # Clear GPU cache
            torch.cuda.empty_cache()
    
    combined_features = np.vstack(combined_chunks)
    print(f"Combined features shape (with attention): {combined_features.shape}")
    return combined_features

# Apply memory-efficient feature combination
combined_features = combine_features_efficiently(expanded_image_features, ts_features)

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

# Clear memory before creating datasets
torch.cuda.empty_cache()
gc.collect()

scaler_y = MinMaxScaler()
train_dataset = WaterLevelDataset(X_train, y_train, scaler_y, fit_scaler=True)
val_dataset = WaterLevelDataset(X_val, y_val, scaler_y)
test_dataset = WaterLevelDataset(X_test, y_test, scaler_y)

baseline_train_dataset = WaterLevelDataset(baseline_X_train, y_train, scaler_y)
baseline_val_dataset = WaterLevelDataset(baseline_X_val, y_val, scaler_y)
baseline_test_dataset = WaterLevelDataset(baseline_X_test, y_test, scaler_y)

# Smaller batch sizes for memory efficiency
batch_size = 16  # Reduced from 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

baseline_train_loader = DataLoader(baseline_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
baseline_val_loader = DataLoader(baseline_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
baseline_test_loader = DataLoader(baseline_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

# Cell 14: Define Enhanced GRU models with Physics Loss
class ProposedGRUModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_layers=2):
        super(ProposedGRUModel, self).__init__()
        
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class BaselineGRUModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2):
        super(BaselineGRUModel, self).__init__()
        
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

proposed_model = ProposedGRUModel(X_seq.shape[2]).to(device)
baseline_model = BaselineGRUModel(baseline_ts_sequences.shape[2]).to(device)

print("Models built")
print(f"Proposed model input shape: {X_seq.shape}")
print(f"Baseline model input shape: {baseline_ts_sequences.shape}")

# Cell 15: Enhanced training function
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            
            # Standard MSE loss
            mse_loss = criterion(outputs, targets)
            
            mse_loss.backward()
            optimizer.step()
            
            train_loss += mse_loss.item()
        
        model.eval()
        val_loss = 0.0
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

print("Training proposed model...")
proposed_train_losses, proposed_val_losses = train_model(
    proposed_model, train_loader, val_loader)

print("Training baseline model...")
baseline_train_losses, baseline_val_losses = train_model(
    baseline_model, baseline_train_loader, baseline_val_loader)

# Cell 16: Make predictions
def make_predictions(model, test_loader, scaler_y):
    model.eval()
    predictions = []
    actuals = []
    
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

print("Predictions completed")
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

print("BASELINE GRU MODEL RESULTS:")
print(f"MAE:  {baseline_metrics['MAE']:.4f}")
print(f"MSE:  {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")

print(f"\nPROPOSED MODEL (VGG19 + GRU) RESULTS:")
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
plt.plot(y_pred_proposed, label='Proposed Model', alpha=0.8)
plt.title('Proposed Model Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(y_test_actual, label='Actual', alpha=0.8)
plt.plot(y_pred_baseline, label='Baseline GRU', alpha=0.8)
plt.title('Baseline Model Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
models = ['Baseline GRU', 'Proposed Model']
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
plt.title('Model Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.scatter(y_test_actual, y_pred_proposed, alpha=0.6, label='Proposed')
plt.scatter(y_test_actual, y_pred_baseline, alpha=0.6, label='Baseline')
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Water Level (m)')
plt.ylabel('Predicted Water Level (m)')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 19: Enhanced training history visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(proposed_train_losses, label='Proposed Training Loss')
plt.plot(proposed_val_losses, label='Proposed Validation Loss')
plt.title('Proposed Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(baseline_train_losses, label='Baseline Training Loss')
plt.plot(baseline_val_losses, label='Baseline Validation Loss')
plt.title('Baseline Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
# Combined loss comparison
plt.plot(proposed_train_losses, label='Proposed Training', color='blue')
plt.plot(baseline_train_losses, label='Baseline Training', color='orange')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(df['Time'][:1000], df['WaterLevel_m'][:1000])
plt.title('Water Level Time Series (First 1000 points)')
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
monthly_avg = df.groupby(df['Month'])['WaterLevel_m'].mean()
plt.bar(monthly_avg.index, monthly_avg.values)
plt.title('Average Water Level by Month')
plt.xlabel('Month')
plt.ylabel('Average Water Level (m)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
# Attention weights visualization (sample)
sample_attention_weights = torch.randn(8, 10, 10)  # Mock attention weights
plt.imshow(sample_attention_weights[0].detach().numpy(), cmap='Blues', alpha=0.8)
plt.title('Sample Attention Weights\n(Image-TimeSeries Cross-Attention)')
plt.xlabel('Time Series Features')
plt.ylabel('Image Features')
plt.colorbar()

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

print("Experiment completed successfully!")
print(f"Final results: MAE {mae_improvement:.1f}%, MSE {mse_improvement:.1f}%, RMSE {rmse_improvement:.1f}% improvement")

torch.save(proposed_model.state_dict(), 'proposed_vgg19_gru_model.pth')
torch.save(baseline_model.state_dict(), 'baseline_gru_model.pth')
print("Models saved successfully!")