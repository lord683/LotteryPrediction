import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

# ---- Data Processing ----
def to_vector(numbers):
    vector = np.zeros(33)
    for number in numbers:
        if number > 0:   # ignore NaN/0
            vector[number-1] = 1
    return vector

# Load data
data = pd.read_csv('data.csv')

X = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].values])
y = np.array([to_vector(numbers) for numbers in data.iloc[:, 1:].shift(-1).fillna(0).values])

# ---- Model ----
inputs = layers.Input(shape=(33,))   # input vector
x = layers.Reshape((1, 33))(inputs)  # expand to 3D (batch, seq_len=1, features=33)

# Transformer block
attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
x = layers.Add()([x, attn_output])
x = layers.LayerNormalization()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Flatten()(x)   # flatten back to 2D
outputs = layers.Dense(33, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- Training ----
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=True,
                             monitor='val_loss', mode='min')

history = model.fit(X, y, batch_size=64, epochs=50, validation_split=0.1, callbacks=[checkpoint])

# ---- Prediction ----
current_numbers = [2, 4, 6, 8, 10, 12, 14]
current_vector = to_vector
