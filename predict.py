import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

VECTOR_SIZE = 49
TOP_N = 6  # main numbers

# =====================
# Helpers
# =====================
def to_vector(numbers):
    vector = np.zeros(VECTOR_SIZE)
    for number in numbers:
        if 0 < number <= VECTOR_SIZE:
            vector[number-1] = 1
    return vector

def vector_to_numbers(vector, top_n=TOP_N):
    indices = np.argsort(vector)[::-1][:top_n]
    return [i+1 for i in indices]

def compute_hot_numbers(data, draw_type, window=10):
    """Compute frequency of numbers in the last `window` draws."""
    subset = data[data['draw_type'] == draw_type].tail(window)
    counts = np.zeros(VECTOR_SIZE)
    for _, row in subset.iterrows():
        numbers = row[2:8].values
        for num in numbers:
            if 0 < num <= VECTOR_SIZE:
                counts[num-1] += 1
    # Normalize to [0,1]
    return counts / counts.max() if counts.max() > 0 else counts

def predict_draw(model, latest_numbers, hot_numbers=None, top_n=TOP_N):
    """Predict next draw and combine with hot numbers if provided"""
    input_vector = to_vector(latest_numbers)
    predicted_vector = model.predict(np.array([input_vector]))[0]

    if hot_numbers is not None:
        # Combine model prediction with hot number weighting
        combined_vector = 0.7 * predicted_vector + 0.3 * hot_numbers
    else:
        combined_vector = predicted_vector

    main_numbers = vector_to_numbers(combined_vector, top_n)
    bonus_number = np.argmax(combined_vector) + 1
    return main_numbers, bonus_number, combined_vector

def show_top_numbers(vector, top_n=10):
    prob_list = [(i+1, round(vector[i],3)) for i in range(len(vector))]
    prob_list.sort(key=lambda x: x[1], reverse=True)
    return prob_list[:top_n]

# =====================
# Load Models
# =====================
model_teatime = load_model("model_teatime.h5")
model_lunchtime = load_model("model_lunchtime.h5")

# =====================
# Load Data
# =====================
data = pd.read_csv("data.csv")
latest_teatime = data[data["draw_type"] == "teatime"].iloc[-1, 2:8].values
latest_lunchtime = data[data["draw_type"] == "lunchtime"].iloc[-1, 2:8].values

hot_teatime = compute_hot_numbers(data, "teatime", window=10)
hot_lunchtime = compute_hot_numbers(data, "lunchtime", window=10)

# =====================
# Predict
# =====================
teatime_main, teatime_bonus, teatime_vector = predict_draw(model_teatime, latest_teatime, hot_teatime)
lunchtime_main, lunchtime_bonus, lunchtime_vector = predict_draw(model_lunchtime, latest_lunchtime, hot_lunchtime)

# =====================
# Print Results
# =====================
print("Latest Teatime:", latest_teatime)
print("Predicted Teatime next numbers:", teatime_main, "| Bonus:", teatime_bonus)
print("Top 10 likely Teatime numbers with hot weighting:", show_top_numbers(teatime_vector))

print("\nLatest Lunchtime:", latest_lunchtime)
print("Predicted Lunchtime next numbers:", lunchtime_main, "| Bonus:", lunchtime_bonus)
print("Top 10 likely Lunchtime numbers with hot weighting:", show_top_numbers(lunchtime_vector))

# =====================
# Save to File
# =====================
with open("predictions.txt", "w") as f:
    f.write("=== Teatime ===\n")
    f.write("Latest draw: " + ", ".join(map(str, latest_teatime)) + "\n")
    f.write("Predicted next numbers: " + ", ".join(map(str, teatime_main)) + " | Bonus: " + str(teatime_bonus) + "\n")
    f.write("Top 10 likely numbers with hot weighting:\n")
    for n,p in show_top_numbers(teatime_vector):
        f.write(f"{n}: {p}\n")

    f.write("\n=== Lunchtime ===\n")
    f.write("Latest draw: " + ", ".join(map(str, latest_lunchtime)) + "\n")
    f.write("Predicted next numbers: " + ", ".join(map(str, lunchtime_main)) + " | Bonus: " + str(lunchtime_bonus) + "\n")
    f.write("Top 10 likely numbers with hot weighting:\n")
    for n,p in show_top_numbers(lunchtime_vector):
        f.write(f"{n}: {p}\n")

print("\n✅ Predictions saved to predictions.txt")
