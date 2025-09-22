# Cell 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
warnings.filterwarnings('ignore')

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
interpolated_images = interpolated_images.reshape(-1, 320, 320, 1)
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

# Cell 6: Build VGG19 feature extractor
def build_vgg19_extractor(input_shape=(320, 320, 1), output_features=1024):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(output_features, activation='relu')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

vgg19_model = build_vgg19_extractor()
print("VGG19 model built")

# Cell 7: Extract image features
image_features = vgg19_model.predict(interpolated_images, batch_size=4, verbose=1)
print(f"Image features shape: {image_features.shape}")

# Cell 8: Build time series feature extractor
def build_ts_extractor(input_dim, output_features=1024):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(output_features, activation='relu')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

feature_columns = ['Inflow_m3s', 'TotalDischarge_m3s', 'IsFloodSeason', 
                  'WaterLevel_lag1', 'Inflow_lag1', 'WaterLevel_lag2', 
                  'Inflow_lag2', 'WaterLevel_lag3', 'Inflow_lag3']

ts_feature_data = df[feature_columns].values
ts_scaler = StandardScaler()
ts_feature_data_scaled = ts_scaler.fit_transform(ts_feature_data)

ts_model = build_ts_extractor(len(feature_columns))
ts_features = ts_model.predict(ts_feature_data_scaled, batch_size=32, verbose=1)
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

# Cell 11: Create sequences for SARIMAX
def create_sequences(features, labels, time_steps=4):
    X, y = [], []
    
    for i in range(time_steps, len(features)):
        X.append(features[i-time_steps:i])
        y.append(labels[i])
    
    X = np.array(X)
    y = np.array(y)
    X_flat = X.reshape(X.shape[0], -1)
    
    return X_flat, y

labels = df['WaterLevel_m'].values
min_len = min(len(combined_features), len(labels))
combined_features = combined_features[:min_len]
labels = labels[:min_len]

X_seq, y_seq = create_sequences(combined_features, labels, time_steps=4)
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

# Cell 13: Find optimal SARIMAX parameters
def find_optimal_sarimax_params(endog, exog=None, max_p=3, max_q=3, max_P=2, max_Q=2):
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    try:
                        model = SARIMAX(endog, exog=exog, order=(p, 1, q), 
                                       seasonal_order=(P, 1, Q, 12),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                        fitted_model = model.fit(disp=False)
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, 1, q, P, 1, Q, 12)
                    except:
                        continue
    
    return best_params, best_aic

print("Finding optimal parameters for proposed model...")
proposed_params, proposed_aic = find_optimal_sarimax_params(y_train, X_train)
print(f"Proposed model best params: {proposed_params} with AIC: {proposed_aic:.4f}")

print("Finding optimal parameters for baseline model...")
baseline_features = ts_feature_data_scaled[:len(y_train)]
baseline_params, baseline_aic = find_optimal_sarimax_params(y_train, baseline_features)
print(f"Baseline model best params: {baseline_params} with AIC: {baseline_aic:.4f}")

# Cell 14: Train SARIMAX models
p, d, q, P, D, Q, s = proposed_params
proposed_model = SARIMAX(y_train, exog=X_train, order=(p, d, q), 
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False, enforce_invertibility=False)
fitted_proposed = proposed_model.fit(disp=False)

p_b, d_b, q_b, P_b, D_b, Q_b, s_b = baseline_params
baseline_model = SARIMAX(y_train, exog=baseline_features, order=(p_b, d_b, q_b), 
                        seasonal_order=(P_b, D_b, Q_b, s_b),
                        enforce_stationarity=False, enforce_invertibility=False)
fitted_baseline = baseline_model.fit(disp=False)

print("Models trained successfully")

# Cell 15: Make predictions
y_pred_proposed = fitted_proposed.forecast(steps=len(X_test), exog=X_test)
baseline_test_features = ts_feature_data_scaled[len(y_train):len(y_train)+len(y_test)]
y_pred_baseline = fitted_baseline.forecast(steps=len(y_test), exog=baseline_test_features)

print("Predictions completed")

# Cell 16: Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

proposed_metrics = calculate_metrics(y_test, y_pred_proposed)
baseline_metrics = calculate_metrics(y_test, y_pred_baseline)

mae_improvement = (baseline_metrics['MAE'] - proposed_metrics['MAE']) / baseline_metrics['MAE'] * 100
mse_improvement = (baseline_metrics['MSE'] - proposed_metrics['MSE']) / baseline_metrics['MSE'] * 100
rmse_improvement = (baseline_metrics['RMSE'] - proposed_metrics['RMSE']) / baseline_metrics['RMSE'] * 100

# Cell 17: Display results
print("BASELINE SARIMAX MODEL RESULTS:")
print(f"MAE:  {baseline_metrics['MAE']:.4f}")
print(f"MSE:  {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")

print(f"\nPROPOSED MODEL (VGG19 + SARIMAX) RESULTS:")
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
plt.plot(y_test, label='Actual', alpha=0.8)
plt.plot(y_pred_proposed, label='Proposed Model', alpha=0.8)
plt.title('Proposed Model Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(y_test, label='Actual', alpha=0.8)
plt.plot(y_pred_baseline, label='Baseline SARIMAX', alpha=0.8)
plt.title('Baseline Model Predictions')
plt.xlabel('Time')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
models = ['Baseline SARIMAX', 'Proposed Model']
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
plt.scatter(y_test, y_pred_proposed, alpha=0.6, label='Proposed')
plt.scatter(y_test, y_pred_baseline, alpha=0.6, label='Baseline')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Water Level (m)')
plt.ylabel('Predicted Water Level (m)')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 19: Additional analysis
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['Time'][:1000], df['WaterLevel_m'][:1000])
plt.title('Water Level Time Series (First 1000 points)')
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
monthly_avg = df.groupby(df['Month'])['WaterLevel_m'].mean()
plt.bar(monthly_avg.index, monthly_avg.values)
plt.title('Average Water Level by Month')
plt.xlabel('Month')
plt.ylabel('Average Water Level (m)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
flood_data = df[df['IsFloodSeason'] == 1]['WaterLevel_m']
dry_data = df[df['IsFloodSeason'] == 0]['WaterLevel_m']
plt.hist(flood_data, alpha=0.7, label='Flood Season', bins=30)
plt.hist(dry_data, alpha=0.7, label='Dry Season', bins=30)
plt.title('Water Level Distribution by Season')
plt.xlabel('Water Level (m)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
sample_images = interpolated_images[:4].reshape(4, 320, 320)
for i in range(4):
    plt.subplot(2, 4, 5 + i)
    plt.imshow(sample_images[i], cmap='gray')
    plt.title(f'Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Cell 20: Save results
results = {
    'proposed_metrics': proposed_metrics,
    'baseline_metrics': baseline_metrics,
    'improvements': {
        'MAE': mae_improvement,
        'MSE': mse_improvement,
        'RMSE': rmse_improvement
    },
    'predictions': {
        'actual': y_test,
        'proposed': y_pred_proposed,
        'baseline': y_pred_baseline
    }
}

print("Experiment completed successfully!")
print(f"Final results: MAE {mae_improvement:.1f}%, MSE {mse_improvement:.1f}%, RMSE {rmse_improvement:.1f}% improvement")