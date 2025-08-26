import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

VECTOR_SIZE = 49  # Total numbers in the lottery

# =====================
# Helpers
# =====================
def to_vector(numbers, vector_size=VECTOR_SIZE):
    vector = np.zeros(vector_size)
    for number in numbers:
        if 0 < number <= vector_size:
            vector[number-1] = 1
    return vector

def build_model():
    inputs = layers.Input(shape=(VECTOR_SIZE,))
    x = layers.Reshape((1, VECTOR_SIZE))(inputs)
    
    # Transformer block
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(VECTOR_SIZE, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =====================
# Load data
# =====================
data = pd.read_csv("data.csv")

for draw_type in ["teatime", "lunchtime"]:
    print(f"\nTraining model for {draw_type} draws...")

    subset = data[data["draw_type"] == draw_type].reset_index(drop=True)
    X = np.array([to_vector(row[2:8]) for _, row in subset.iterrows()])
    y = np.array([to_vector(row[2:8]) for _, row in subset.shift(-1).dropna().iterrows()])

    model = build_model()
    model.summary()

    checkpoint = ModelCheckpoint(
        f"model_{draw_type}.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    model.fit(
        X, y,
        batch_size=64,
        epochs=50,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    print(f"✅ {draw_type.capitalize()} model trained and saved as model_{draw_type}.h5")
