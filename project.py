import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import re
import time

# Load the dataset
df = pd.read_csv('telemetry_log.csv')

# Process the focus column which contains space-separated values
def process_focus_column(focus_str):
    if pd.isna(focus_str):
        return [-1.0] * 5
    try:
        values = [float(x) for x in str(focus_str).split()]
        # Ensure consistent length by padding or truncating
        if len(values) < 5:
            values.extend([-1.0] * (5 - len(values)))
        return values[:5]  # Take only first 5 values
    except:
        return [-1.0] * 5  # Default value if parsing fails

# Process opponent data columns
def process_opponent_data(data_str):
    if pd.isna(data_str):
        return [-1.0] * 4  # Default values for missing data
    try:
        # Expected format of opponent data: distance speedX speedY speedZ
        values = [float(x) for x in str(data_str).split()]
        if len(values) < 4:
            values.extend([-1.0] * (4 - len(values)))
        return values[:4]  # Take only first 4 values
    except:
        return [-1.0] * 4

# Apply focus processing
if 'focus' in df.columns:
    focus_data = df['focus'].apply(process_focus_column)
    for i in range(5):
        df[f'focus_{i+1}'] = focus_data.apply(lambda x: x[i])

# Process opponent data
opponent_data_columns = [col for col in df.columns if col.endswith('_data')]
for col in opponent_data_columns:
    opponent_num = col.split('_')[1]
    data = df[col].apply(process_opponent_data)
    for i, metric in enumerate(['dist', 'speedX', 'speedY', 'speedZ']):
        df[f'opponent_{opponent_num}_{metric}'] = data.apply(lambda x: x[i])

# Process opponent position columns
opponent_pos_columns = [col for col in df.columns if col.endswith('_pos')]
for col in opponent_pos_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define our feature columns
# Base car state features
car_state_features = [
    'step', 'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced',
    'fuel', 'lastLapTime', 'racePos', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos'
]

# Track features (from focus columns)
track_features = [f'focus_{i+1}' for i in range(5) if f'focus_{i+1}' in df.columns]

# Opponent features
opponent_features = []
for i in range(1, 6):  # Assuming 5 opponents
    # Add position
    pos_col = f'opponent_{i}_pos'
    if pos_col in df.columns:
        opponent_features.append(pos_col)
    
    # Add processed data columns
    for metric in ['dist', 'speedX', 'speedY', 'speedZ']:
        col = f'opponent_{i}_{metric}'
        if col in df.columns:
            opponent_features.append(col)

# Combine all feature columns
all_feature_columns = car_state_features + track_features + opponent_features

# Define label columns - all control outputs
label_columns = ['accel', 'brake', 'gear', 'steer', 'clutch', 'focus', 'meta']

# Convert all columns to numeric
for col in all_feature_columns + label_columns:
    if col in df.columns and col != 'focus':  # Skip the original focus column
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare the focus label (original focus column) if it's being predicted
if 'focus' in label_columns and 'focus' in df.columns:
    # For prediction, we'll use a single number representing focus direction
    # A value between 0-1 where 0 means far left, 0.5 means center, 1 means far right
    df['focus_target'] = df['focus'].apply(
        lambda x: 0.5 if pd.isna(x) else 
        np.mean([i/(len(process_focus_column(x))-1) for i in range(len(process_focus_column(x)))])
    )
    df['focus'] = df['focus_target']

# Drop rows with NaN values in important columns
df = df.dropna(subset=[col for col in car_state_features if col in df.columns])

# Fill remaining NaN values with sensible defaults
df.fillna({
    'meta': 0,  # Default meta action
    'focus': 0.5,  # Default focus (center)
}, inplace=True)

# Select only columns that are actually in the dataframe
available_features = [col for col in all_feature_columns if col in df.columns]
available_labels = [col for col in label_columns if col in df.columns]

print(f"Using {len(available_features)} features: {available_features}")
print(f"Predicting {len(available_labels)} labels: {available_labels}")

# Select features and labels
X = df[available_features].values
y = df[available_labels].values

# Train-test split - 90% training, 10% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine number of unique gears
num_gears = int(df['gear'].max()) + 1

# Neural Network with optimized architecture
def build_model(input_dim, num_gears):
    # Use Keras Functional API for multi-output model
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared layers
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Specialized layers for different control types
    
    # Steering branch (needs extra precision)
    steer_branch = layers.Dense(64, activation='relu')(x)
    steer_branch = layers.Dense(32, activation='relu')(steer_branch)
    steer_output = layers.Dense(1, activation='tanh', name='steer')(steer_branch)
    
    # Speed control branch (throttle/brake)
    speed_branch = layers.Dense(64, activation='relu')(x)
    accel_output = layers.Dense(1, activation='sigmoid', name='accel')(speed_branch)
    brake_output = layers.Dense(1, activation='sigmoid', name='brake')(speed_branch)
    
    # Gear branch (classification task)
    gear_branch = layers.Dense(64, activation='relu')(x)
    gear_output = layers.Dense(num_gears, activation='softmax', name='gear')(gear_branch)
    
    # Other controls
    clutch_output = layers.Dense(1, activation='sigmoid', name='clutch')(x)
    focus_output = layers.Dense(1, activation='sigmoid', name='focus')(x)
    meta_output = layers.Dense(1, activation='sigmoid', name='meta')(x)
    
    # Create model with all outputs
    model = keras.Model(
        inputs=inputs, 
        outputs=[
            accel_output, brake_output, gear_output, 
            steer_output, clutch_output, focus_output, meta_output
        ]
    )
    
    # Compile with appropriate loss functions and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'accel': 'mse',
            'brake': 'mse',
            'gear': 'sparse_categorical_crossentropy',
            'steer': 'mse',
            'clutch': 'mse',
            'focus': 'mse',
            'meta': 'binary_crossentropy'
        },
        loss_weights={
            'accel': 2.0,    # Prioritize acceleration control
            'brake': 2.0,    # Prioritize braking control
            'gear': 1.5,     # Important but slightly less than accel/brake
            'steer': 3.0,    # Most important for control
            'clutch': 0.5,   # Less critical
            'focus': 0.5,    # Less critical
            'meta': 0.5      # Less critical
        },
        metrics={
            'accel': 'mae',
            'brake': 'mae',
            'gear': 'accuracy',
            'steer': 'mae',
            'clutch': 'mae',
            'focus': 'mae',
            'meta': 'accuracy'
        }
    )
    return model

# Prepare labels in dictionary format for multi-output model
y_train_dict = {
    'accel': y_train[:, 0],
    'brake': y_train[:, 1],
    'gear': y_train[:, 2].astype(int),
    'steer': y_train[:, 3],
    'clutch': y_train[:, 4],
    'focus': y_train[:, 5],
    'meta': y_train[:, 6]
}

y_test_dict = {
    'accel': y_test[:, 0],
    'brake': y_test[:, 1],
    'gear': y_test[:, 2].astype(int),
    'steer': y_test[:, 3],
    'clutch': y_test[:, 4],
    'focus': y_test[:, 5],
    'meta': y_test[:, 6]
}

# Build and train the model
model = build_model(X_train_scaled.shape[1], num_gears)
model.summary()

# Callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_torcs_driver.keras', 
        save_best_only=True, 
        monitor='val_loss'
    )
]

# Train with appropriate batch size and epochs
print("\nTraining model...")
start_time = time.time()

history = model.fit(
    X_train_scaled, 
    y_train_dict,
    validation_data=(X_test_scaled, y_test_dict),
    epochs=100,  # Maximum epochs (early stopping will likely trigger before this)
    batch_size=64,  # Larger batch size for faster training
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate on test set
print("\nEvaluation Results:")
eval_results = model.evaluate(X_test_scaled, y_test_dict, verbose=0)
for name, val in zip(model.metrics_names, eval_results):
    print(f"{name}: {val:.4f}")

# Calculate accuracy metrics for each control output
def calculate_regression_accuracy(true_values, pred_values, name):
    true_values = np.array(true_values).flatten()
    pred_values = np.array(pred_values).flatten()
    
    mse = mean_squared_error(true_values, pred_values)
    rmse = np.sqrt(mse)
    
    # For steering, use smaller threshold since it's more sensitive
    if name == 'steer':
        max_error = 2.0  # Range is -1 to 1, so max error is 2
        threshold = 0.1  # 10% of total range is considered "correct"
    else:
        max_error = 1.0  # Most controls range from 0-1
        threshold = 0.15  # 15% of range is considered "correct"
    
    # Calculate percentage of predictions within acceptable threshold
    within_threshold = np.mean(np.abs(true_values - pred_values) < threshold) * 100
    
    # Calculate normalized accuracy (higher is better)
    normalized_accuracy = 100 * (1 - rmse / max_error)
    
    return max(0, normalized_accuracy), within_threshold

# For gear classification, calculate percentage accuracy
def calculate_classification_accuracy(true_values, pred_values):
    if isinstance(pred_values, np.ndarray) and pred_values.ndim > 1:
        pred_classes = np.argmax(pred_values, axis=1)
    else:
        pred_classes = np.round(pred_values).astype(int)
    return accuracy_score(true_values, pred_classes) * 100

# Predict on test set
print("\nGenerating predictions for accuracy assessment...")
y_pred_dict = model.predict(X_test_scaled, verbose=0)


print("\nGenerating predictions for accuracy assessment...")
output_names = ['accel', 'brake', 'gear', 'steer', 'clutch', 'focus', 'meta']
y_pred_list = model.predict(X_test_scaled, verbose=0)
y_pred_dict = dict(zip(output_names, y_pred_list))

print("\nAccuracy Metrics:")
print("-" * 60)
print(f"{'Control':<10} | {'Norm. Accuracy':<15} | {'Within Threshold %':<20}")
print("-" * 60)

accuracies = {}
threshold_accuracies = {}

# For regression outputs
for control in ['accel', 'brake', 'steer', 'clutch', 'focus']:
    norm_acc, thresh_acc = calculate_regression_accuracy(
        y_test_dict[control], y_pred_dict[control], control
    )
    accuracies[control] = norm_acc
    threshold_accuracies[control] = thresh_acc
    print(f"{control:<10} | {norm_acc:>13.2f}% | {thresh_acc:>18.2f}%")

# For classification outputs
for control in ['gear', 'meta']:
    acc = calculate_classification_accuracy(y_test_dict[control], y_pred_dict[control])
    accuracies[control] = acc
    threshold_accuracies[control] = acc
    print(f"{control:<10} | {acc:>13.2f}% | {'N/A':>18}")

print("-" * 60)
# Test inference speed
print("\nTesting inference speed...")
# Generate 1000 random inputs for speed testing
random_inputs = np.random.randn(1000, X_train_scaled.shape[1])
random_inputs_scaled = scaler.transform(random_inputs)

start_time = time.time()
_ = model.predict(random_inputs_scaled, verbose=0)
inference_time = (time.time() - start_time) / 1000
print(f"Average inference time per input: {inference_time*1000:.2f} ms")

# Plot training history
plt.figure(figsize=(15, 10))

# Plot overall loss
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Overall Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot selected metrics
metrics_to_plot = [
    ('steer_mae', 'Steering MAE'),
    ('accel_mae', 'Acceleration MAE'),
    ('brake_mae', 'Brake MAE'),
    ('gear_accuracy', 'Gear Accuracy'),
    ('meta_accuracy', 'Meta Action Accuracy')
]

for i, (metric, title) in enumerate(metrics_to_plot):
    if metric in history.history:
        plt.subplot(2, 3, i+2)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()

plt.tight_layout()
plt.savefig('torcs_training_history.png')
plt.show()

# Create a real-time inference function that takes sensor data string input
def preprocess_sensor_data(sensor_data_str):
    # Parse the sensor data string into a structured format
    sensor_data = {}
    parts = re.findall(r'\(([^)]+)\)', sensor_data_str)
    
    for part in parts:
        if ' ' in part:
            key, values = part.split(' ', 1)
            try:
                # Try to convert to a list of floats
                values = [float(v) for v in values.split()]
                sensor_data[key] = values
            except:
                sensor_data[key] = values
        else:
            sensor_data[part] = 0  # Handle edge case of empty values
    
    # Process focus data if present
    if 'focus' in sensor_data:
        focus_values = process_focus_column(sensor_data['focus'])
        for i, val in enumerate(focus_values):
            sensor_data[f'focus_{i+1}'] = val
    
    # Process opponent data if present
    for i in range(1, 6):
        opponent_key = f'opponent_{i}_data'
        if opponent_key in sensor_data:
            opp_values = process_opponent_data(sensor_data[opponent_key])
            for j, metric in enumerate(['dist', 'speedX', 'speedY', 'speedZ']):
                sensor_data[f'opponent_{i}_{metric}'] = opp_values[j]
    
    # Extract feature values in the same order as training
    feature_values = []
    for feature in available_features:
        # Handle special cases for composite features
        if feature in sensor_data:
            if isinstance(sensor_data[feature], list):
                feature_values.append(sensor_data[feature][0] if sensor_data[feature] else 0)
            else:
                feature_values.append(float(sensor_data[feature]) if sensor_data[feature] else 0)
        else:
            # If feature not in sensor data, use 0 as default
            feature_values.append(0)
    
    return np.array([feature_values])

def predict_controls(sensor_data_str):
    start_time = time.time()
    
    # Preprocess the sensor data string
    input_array = preprocess_sensor_data(sensor_data_str)
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    preds = model.predict(input_scaled, verbose=0)
    
    # Extract prediction results
    accel = float(preds[0][0][0])
    brake = float(preds[1][0][0])
    gear = int(np.argmax(preds[2][0]))
    steer = float(preds[3][0][0])
    clutch = float(preds[4][0][0])
    focus = float(preds[5][0][0])
    meta = int(round(float(preds[6][0][0])))
    
    # Scale focus to appropriate format (5 space-separated values)
    # Focus is a direction value between 0-1, convert to 5 target points
    # with highest value at the target direction
    focus_values = [0.1] * 5
    focus_target = int(min(4, round(focus * 4)))  # Convert to index 0-4
    focus_values[focus_target] = 0.9
    focus_str = " ".join([f"{v:.1f}" for v in focus_values])
    
    # Format the output as required
    result = f"(accel {accel:.3f})(brake {brake:.3f})(gear {gear})(steer {steer:.3f})(clutch {clutch:.3f})(focus {focus_str})(meta {meta})"
    
    inference_time = time.time() - start_time
    # Uncomment for debugging
    # print(f"Inference time: {inference_time*1000:.2f}ms")
    
    return result

# Example prediction
sample = "(angle 0.00088102)(curLapTime -0.682)(damage 0)(distFromStart 1037.93)(distRaced 0)(fuel 93.9992)(gear 0)(lastLapTime 0)(racePos 2)(rpm 2945.14)(speedX -0.0179197)(speedY -0.000773841)(speedZ -0.0008619)(trackPos 0.333339)"
print("\nSample Input:")
print(sample)
print("\nPredicted Controls:")
print(predict_controls(sample))

# Create a TensorFlow Lite model for even faster inference
print("\nCreating TensorFlow Lite model for faster inference...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open('torcs_driver_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save the model and scaler
model_data = {
    'feature_columns': available_features,
    'label_columns': available_labels,
    'scaler': scaler,
    'num_gears': num_gears,
    'accuracies': accuracies,
    'threshold_accuracies': threshold_accuracies,
    'inference_time_ms': inference_time * 1000
}

# Save the full model separately (it's large)
model.save('torcs_driver_model.keras')

# Save the metadata and scaler
with open('torcs_driver_metadata.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModels saved as:")
print("- torcs_driver_model.keras (Full model)")
print("- torcs_driver_model.tflite (Optimized TFLite model)")
print("- torcs_driver_metadata.pkl (Metadata and scaler)")

# Save metadata to a text file for reference
with open('torcs_model_metadata.txt', 'w') as f:
    f.write("TORCS Driver Model Metadata\n")
    f.write("==========================\n\n")
    f.write(f"Features ({len(available_features)}):\n")
    for feature in available_features:
        f.write(f"- {feature}\n")
    f.write(f"\nLabels ({len(available_labels)}):\n")
    for label in available_labels:
        f.write(f"- {label}\n")
    f.write("\nAccuracy Results:\n")
    for feature, accuracy in accuracies.items():
        f.write(f"- {feature.capitalize()}: {accuracy:.2f}% (within threshold: {threshold_accuracies.get(feature, 'N/A')}%)\n")
    f.write(f"\nAverage inference time: {inference_time*1000:.2f} ms\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")