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
    vector = np.zeros(max_num)
    for n in numbers:
        if n > 0:
            vector[n-1] = 1
    return vector

# Convert the historical data into one-hot encoded vectors
X = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].values])
y = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].shift(-1).fillna(0).values])

# --- Reshape data for Transformer ---
X = X.reshape(X.shape[0], X.shape[1], 1)  # Transformer requires 3D input (samples, time steps, features)
y = y.reshape(y.shape[0], y.shape[1])

# --- Define Transformer model ---
inputs = layers.Input(shape=(X.shape[1], 1))  # Shape of the input data
transformer_layer = layers.MultiHeadAttention(num_heads=8, key_dim=64)
x = transformer_layer(inputs, inputs)
x = layers.Dropout(0.1)(x)
x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)  # Adding residual connection
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(49, activation='softmax')(x)  # Output layer to predict probabilities for each number

model = models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Checkpoint ---
checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

# --- Train the model ---
model.fit(X, y, batch_size=64, epochs=50, validation_split=0.1, callbacks=[checkpoint])

# --- Predict next draw ---
last_draw_vector = X[-1].reshape(1, -1, 1)  # Use the last draw for prediction
predicted_vector = model.predict(last_draw_vector)[0]  # Predict probabilities for the next draw
predicted_numbers = [i+1 for i, v in enumerate(predicted_vector) if v > 0.5]  # Choose numbers with a probability > 0.5

# --- Save prediction ---
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
with open(OUTPUT_FILE, "a") as f:
    f.write(f"{timestamp} Prediction: {predicted_numbers}\n")

print(f"[+] Prediction saved: {predicted_numbers}")
