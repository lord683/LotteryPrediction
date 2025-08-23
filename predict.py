import pandas as pd

# Load historical data
df = pd.read_csv('data/uk49s_history.csv')

# Simple prediction logic example (last draw numbers)
last_draw = df.iloc[-1, 1:].tolist()  # skip date column
predicted_numbers = last_draw  # For demo, just repeat last draw

# Save predictions to a file
with open("predictions.txt", "w") as f:
    f.write(f"Today's UK49s Prediction: {predicted_numbers}\n")
