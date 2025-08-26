import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

VECTOR_SIZE = 49

# =====================
# Helpers
# =====================
def to_vector(numbers):
    vector = np.zeros(VECTOR_SIZE)
    for number in numbers:
        if 0 < number <= VECTOR_SIZE:
            vector[number-1] = 1
    return vector

def vector_to_numbers(vector, top_n=6):
    indices = np.argsort(vector)[::-1][:top_n]
    return [i+1 for i in indices]

def predict_draw(model, latest_numbers, top_n=6):
    input_vector = to_vector(latest_numbers)
    predicted_vector = model.predict(np.array([input_vector]))[0]
    main_numbers = vector_to_numbers(predicted_vector, top_n)
    bonus_number = np.argmax(predicted_vector) + 1
    return main_numbers, bonus_number, predicted_vector

def show_probabilities(vector, top_n=10):
    prob_list = [(i+1, round(vector[i], 3)) for i in range(len(vector))]
    prob_list.sort(key=lambda x: x[1], reverse=True)
    return prob_list[:top_n]

# =====================
# Load models
# =====================
model_teatime = load_model("model_teatime.h5")
model_lunchtime = load_model("model_lunchtime.h5")

# =====================
# Read latest draws
# =====================
data = pd.read_csv("data.csv")
latest_teatime = data[data["draw_type"] == "teatime"].iloc[-1, 2:8].values
latest_lunchtime = data[data["draw_type"] == "lunchtime"].iloc[-1, 2:8].values

# =====================
# Predictions
# =====================
teatime_main, teatime_bonus, teatime_probs = predict_draw(model_teatime, latest_teatime)
lunchtime_main, lunchtime_bonus, lunchtime_probs = predict_draw(model_lunchtime, latest_lunchtime)

# =====================
# Print Results
# =====================
print("Latest Teatime:", latest_teatime)
print("Predicted Teatime next numbers:", teatime_main, "| Bonus:", teatime_bonus)
print("Top 10 Teatime probabilities:", show_probabilities(teatime_probs))

print("\nLatest Lunchtime:", latest_lunchtime)
print("Predicted Lunchtime next numbers:", lunchtime_main, "| Bonus:", lunchtime_bonus)
print("Top 10 Lunchtime probabilities:", show_probabilities(lunchtime_probs))

# =====================
# Save to file
# =====================
with open("predictions.txt", "w") as f:
    f.write("Latest Teatime: " + ", ".join(map(str, latest_teatime)) + "\n")
    f.write("Predicted Teatime next numbers: " + ", ".join(map(str, teatime_main)) + " | Bonus: " + str(teatime_bonus) + "\n")
    f.write("Top 10 Teatime probabilities:\n")
    for n, p in show_probabilities(teatime_probs):
        f.write(f"{n}: {p}\n")

    f.write("\nLatest Lunchtime: " + ", ".join(map(str, latest_lunchtime)) + "\n")
    f.write("Predicted Lunchtime next numbers: " + ", ".join(map(str, lunchtime_main)) + " | Bonus: " + str(lunchtime_bonus) + "\n")
    f.write("Top 10 Lunchtime probabilities:\n")
    for n, p in show_probabilities(lunchtime_probs):
        f.write(f"{n}: {p}\n")

print("\n✅ Predictions saved to predictions.txt")
