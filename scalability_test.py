import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from LinearSVC import LinearSVC  # Import your implemented SVM
from generate_data import make_classification  # Import dataset generation function
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define dataset configurations
dimensions = [10, 50, 100, 500, 1000]  # Feature dimensions
samples = [500, 1000, 5000, 10000, 100000]  # Number of samples
u = 5  # Range for feature values
seed = 42  # Random seed for reproducibility

scalability_results = []  # Store training times
#accuracy
scalability_acc = []

# Iterate over different dataset sizes
for d in dimensions:
    # XY = pd.read_csv(f"datased_d{d}_n{samples[len(samples)-1]}.csv")
    # X = XY.iloc[:, :-1]
    # Y = XY.iloc[:, -1]

    for n in samples:
        print(f"Generating dataset with d={d}, n={n}...")
        # xsamp = X[:n].values
        # ysamp = Y[:n].values
        
        # Generate dataset
        X_train, X_test, y_train, y_test, _, _, _ = make_classification(d, n, u, seed)
        #X_train, X_test, y_train, y_test = train_test_split(xsamp, ysamp, test_size=0.3, random_state=seed)

        print(f"Training LinearSVC on d={d}, n={n}...")

        # Initialize and train the LinearSVC model
        model = LinearSVC(learning_rate=0.01, epochs=500, regularization_param=0.01)
        
        start_time = time.time()
        model.fit(X_train, y_train)  # Train model
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds.")

        predictions = model.predict(X_test)


        # Store results
        scalability_results.append((d, n, training_time))
        scalability_acc.append((d,n,accuracy_score(y_test, predictions)))

# Convert results to a DataFrame and save
df_scalability = pd.DataFrame(scalability_results, columns=["Dimension (d)", "Samples (n)", "Training Time (s)"])
df_scalability.to_csv("scalability_results.csv", index=False)

df_scalability2 = pd.DataFrame(scalability_acc, columns=["Dimension (d)", "Samples (n)", "Testing Accuracy"])
df_scalability2.to_csv("scalability_acc.csv", index=False)

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

plt.figure(1)
plt.figure(figsize=(8, 6))

for d in dimensions:
    subset = df_scalability2[df_scalability2["Dimension (d)"] == d]
    plt.plot(subset["Samples (n)"], subset["Testing Accuracy"], label=f"d={d}", marker='o')

plt.xlabel("Number of Samples (n)")
plt.ylabel("Testing Accuracy")
plt.title("Scalability of LinearSVC with Increasing d and n")
plt.legend()
plt.grid(True)
plt.show()
