import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from LinearSVC import LinearSVC  # Import your implemented SVM
from generate_data import make_classification  # Import dataset generation function

# Define dataset configurations
dimensions = [10, 50, 100, 500, 1000]  # Feature dimensions
samples = [500, 1000, 5000, 10000, 100000]  # Number of samples
u = 5  # Range for feature values
seed = 42  # Random seed for reproducibility

scalability_results = []  # Store training times

# Iterate over different dataset sizes
for d in dimensions:
    for n in samples:
        print(f"Generating dataset with d={d}, n={n}...")
        
        # Generate dataset
        X_train, X_test, y_train, y_test, _, _, _ = make_classification(d, n, u, seed)

        print(f"Training LinearSVC on d={d}, n={n}...")

        # Initialize and train the LinearSVC model
        model = LinearSVC(learning_rate=0.01, epochs=500, regularization_param=0.01)
        
        start_time = time.time()
        model.fit(X_train, y_train)  # Train model
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds.")

        # Store results
        scalability_results.append((d, n, training_time))

# Convert results to a DataFrame and save
df_scalability = pd.DataFrame(scalability_results, columns=["Dimension (d)", "Samples (n)", "Training Time (s)"])
df_scalability.to_csv("scalability_results.csv", index=False)

# Plot training time trends
plt.figure(figsize=(8, 6))

for d in dimensions:
    subset = df_scalability[df_scalability["Dimension (d)"] == d]
    plt.plot(subset["Samples (n)"], subset["Training Time (s)"], label=f"d={d}", marker='o')

plt.xlabel("Number of Samples (n)")
plt.ylabel("Training Time (seconds)")
plt.title("Scalability of LinearSVC with Increasing d and n")
plt.legend()
plt.grid(True)
plt.show()
