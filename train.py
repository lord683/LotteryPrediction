# train.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import os

CSV_FILE = "history.csv"

# 1️⃣ Load CSV safely
data = pd.read_csv(CSV_FILE, delimiter=",")
print("Columns detected:", data.columns.tolist())

# 2️⃣ Convert numbers to one-hot vectors (size 49)
def to_vector(numbers, max_num=49):
    vec = np.zeros(max_num)
    for n in numbers:
        vec[n-1] = 1
    return vec

# 3️⃣ Prepare dataset for a given draw_type
def prepare_data(draw_type):
    subset = data[data["draw_type"] == draw_type].reset_index(drop=True)
    X = np.array([to_vector(row[["n1","n2","n3","n4","n5","n6"]].values) for idx,row in subset.iterrows()])
    # Predict next draw
    y_raw = subset[["n1","n2","n3","n4","n5","n6"]].shift(-1).fillna(0).astype(int)
    y = np.array([to_vector(row.values) for idx,row in y_raw.iterrows()])
    return X, y

# 4️⃣ Build simple model
def build_model(input_dim=49):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5️⃣ Train and save model for both draw types
for draw_type in ["teatime", "lunchtime"]:
    print(f"Training model for {draw_type} draws...")
    X, y = prepare_data(draw_type)
    if len(X) < 2:
        print(f"Not enough data for {draw_type}, skipping...")
        continue

    model = build_model()
    checkpoint_path = f"model_{draw_type}.weights.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='loss', mode='min')
    
    model.fit(X, y, batch_size=4, epochs=50, callbacks=[checkpoint], verbose=1)
    print(f"✅ {draw_type} model trained and saved: {checkpoint_path}")
