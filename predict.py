import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

VECTOR_SIZE = 49
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

def vector_to_numbers(vector, top_n=TOP_N):
    indices = np.argsort(vector)[::-1][:top_n]
    return [i+1 for i in indices]

def compute_hot_numbers(data, draw_type, window=10):
    subset = data[data['draw_type'] == draw_type].tail(window)
    counts = np.zeros(VECTOR_SIZE)
    for _, row in subset.iterrows():
        nums = row[['n1','n2','n3','n4','n5','n6']].values
        for n in nums:
            if 0 < n <= VECTOR_SIZE:
                counts[n-1] += 1
    return counts / counts.max() if counts.max() > 0 else counts

def predict_draw(model, latest_numbers, hot_numbers=None):
    input_vec = to_vector(latest_numbers)
    pred_vec = model.predict(np.array([input_vec]))[0]
    
    if hot_numbers is not None:
        combined_vec = 0.7 * pred_vec + 0.3 * hot_numbers
    else:
        combined_vec = pred_vec
    
    main_nums = vector_to_numbers(combined_vec, TOP_N)
    bonus_num = np.argmax(combined_vec) + 1
    return main_nums, bonus_num, combined_vec

def show_top_numbers(vector, top_n=10):
    lst = [(i+1, round(vector[i],3)) for i in range(len(vector))]
    lst.sort(key=lambda x: x[1], reverse=True)
    return lst[:top_n]

# =====================
# Load models
# =====================
model_teatime = load_model("model_teatime.h5")
model_lunchtime = load_model("model_lunchtime.h5")

# =====================
# Load latest draws
# =====================
data = pd.read_csv("data.csv")
latest_teatime = data[data["draw_type"] == "teatime"].iloc[-1][['n1','n2','n3','n4','n5','n6']].values
latest_lunchtime = data[data["draw_type"] == "lunchtime"].iloc[-1][['n1','n2','n3','n4','n5','n6']].values

hot_teatime = compute_hot_numbers(data, "teatime", window=10)
hot_lunchtime = compute_hot_numbers(data, "lunchtime", window=10)

# =====================
# Predict
# =====================
t_main, t_bonus, t_vec = predict_draw(model_teatime, latest_teatime, hot_teatime)
l_main, l_bonus, l_vec = predict_draw(model_lunchtime, latest_lunchtime, hot_lunchtime)

# =====================
# Print
# =====================
print("Latest Teatime:", latest_teatime)
print("Predicted Teatime:", t_main, "| Bonus:", t_bonus)
print("Top 10 likely Teatime numbers:", show_top_numbers(t_vec))

print("\nLatest Lunchtime:", latest_lunchtime)
print("Predicted Lunchtime:", l_main, "| Bonus:", l_bonus)
print("Top 10 likely Lunchtime numbers:", show_top_numbers(l_vec))

# =====================
# Save to file
# =====================
with open("predictions.txt", "w") as f:
    f.write("=== Teatime ===\n")
    f.write("Latest draw: " + ", ".join(map(str, latest_teatime)) + "\n")
    f.write("Predicted next numbers: " + ", ".join(map(str, t_main)) + " | Bonus: " + str(t_bonus) + "\n")
    f.write("Top 10 likely numbers with hot weighting:\n")
    for n,p in show_top_numbers(t_vec):
        f.write(f"{n}: {p}\n")
    
    f.write("\n=== Lunchtime ===\n")
    f.write("Latest draw: " + ", ".join(map(str, latest_lunchtime)) + "\n")
    f.write("Predicted next numbers: " + ", ".join(map(str, l_main)) + " | Bonus: " + str(l_bonus) + "\n")
    f.write("Top 10 likely numbers with hot weighting:\n")
    for n,p in show_top_numbers(l_vec):
        f.write(f"{n}: {p}\n")

print("\n✅ Predictions saved to predictions.txt")
