import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

VECTOR_SIZE = 49  # Total lottery numbers
TOP_N = 6

# =====================
# Helpers
# =====================
def to_vector(numbers):
    vec = np.zeros(VECTOR_SIZE)
    for n in numbers:
        if 0 < n <= VECTOR_SIZE:
            vec[n-1] = 1
    return vec

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
# Load Data
# =====================
data = pd.read_csv("data.csv")

for draw_type in ["teatime", "lunchtime"]:
    print(f"\nTraining model for {draw_type} draws...")
    subset = data[data["draw_type"] == draw_type].reset_index(drop=True)
    
    # Input vectors
    X = np.array([to_vector(row[['n1','n2','n3','n4','n5','n6']].values) for _, row in subset.iterrows()])
    # Target = next draw
    y = np.array([to_vector(row[['n1','n2','n3','n4','n5','n6']].values) for _, row in subset.shift(-1).dropna().iterrows()])
    
    model = build_model()
    model.summary()
    
    checkpoint = ModelCheckpoint(
        f"model_{draw_type}.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    
    model.fit(
        X[:len(y)], y,  # match length because last row has no next draw
        batch_size=64,
        epochs=50,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    
    print(f"✅ {draw_type.capitalize()} model trained and saved as model_{draw_type}.h5")
