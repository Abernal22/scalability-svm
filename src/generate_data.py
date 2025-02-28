import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import time

def make_classification(d, n, u=5, seed=42, save_to_file = False):

    np.random.seed(seed)
    
    #Generate a random normal vector that defines the hyperplane
    a = np.random.randn(d)

    #generate random samples in [-u, u]^d using uniform
    X = np.random.uniform(-u, u, (n,d))

    #compute the dot product a^T X and assign lables based on the sign (neg or pos)
    y = np.sign(X @ a)

    #split testing and training data 70% for training and 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    #save datset to CSV file
    if save_to_file:
        dataset_filename = f"datased_d{d}_n{n}.csv"
        hyperplane_filename = f"hyperplane_d{d}_n{n}.csv"

        #save data set
        df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
        df["label"] = y
        df.to_csv(dataset_filename, index=False)

        #save hyper plane vector
        pd.DataFrame(a.reshape(1, -1), columns=[f"a_{i+1}" for i in range(d)]).to_csv(hyperplane_filename, index=False)

        print(f"Dataset saved to {dataset_filename}")
        print(f"Hyperplane vector saved to {hyperplane_filename}")


    return X_train, X_test, y_train, y_test, X, y, a

#generate 2d data set with n=100 samples
if __name__ == "__main__":
  d = 2
  n = 100
  u = 5
  X_train, X_test, y_train, y_test, X, y, a = make_classification(d, n, u, save_to_file = True)

  #plot the data set
  plt.figure(figsize=(8, 6))
  plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='red', label='Class -1')
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')

  #plot decision hyperplane (a^T x = 0)
  x_vals = np.linspace(-u, u, 100)
  y_vals = - (a[0] * x_vals) / a[1]  # Since a1*x + a2*y = 0 => y = - (a1/a2) * x
  plt.plot(x_vals, y_vals, 'k--', label="Hyperplane")
  plt.legend()
  plt.title("Linearly Separable Data (d=2, n=100)")
  plt.show()
