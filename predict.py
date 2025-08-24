import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "uk49s_history.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.h5")
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions.txt")

# --- Load data ---
data = pd.read_csv(DATA_FILE)

# --- One-hot encode numbers ---
def to_vector(numbers, max_num=49):
    """Convert the list of numbers to a one-hot encoded vector"""
    vector = np.zeros(max_num, dtype=int)
    for n in numbers:
        try:
            n = int(n)
            if 1 <= n <= max_num:  # only valid lottery numbers
                vector[n-1] = 1
        except (ValueError, TypeError):
            continue  # skip invalid entries
    return vector

# --- Prepare dataset ---
X = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].values])
y = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].shift(-1).fillna(0).values])

# Remove last row if shapes don’t match
min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]

# --- Reshape for Transformer ---
X = X.reshape(X.shape[0], X.shape[1], 1)  # samples, timesteps, features
y = y.reshape(y.shape[0], y.shape[1])

# --- Define Transformer-like model ---
inputs = layers.Input(shape=(X.shape[1], 1))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
x = layers.Dropout(0.1)(attn)
x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)  # residual connection
x = layers.Flatten()(x)  # flatten sequence
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(49, activation='sigmoid')(x)  # sigmoid for multi-label classification

model = models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Checkpoint ---
checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

# --- Train the model ---
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[checkpoint])

# --- Predict next draw ---
last_draw_vector = X[-1].reshape(1, X.shape[1], 1)
predicted_vector = model.predict(last_draw_vector)[0]

# Pick top 6 numbers instead of probability > 0.5
predicted_numbers = np.argsort(predicted_vector)[-6:] + 1
predicted_numbers = sorted(predicted_numbers.tolist())

# --- Save prediction ---
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
with open(OUTPUT_FILE, "a") as f:
    f.write(f"{timestamp} Prediction: {predicted_numbers}\n")

print(f"[+] Prediction saved: {predicted_numbers}")
