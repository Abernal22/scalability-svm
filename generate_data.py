import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_classification(d, n, u, seed=42):

    np.random.seed(seed)
    
    #Generate a random normal vector that defines the hyperplane
    a = np.random.randn(d)

    #generate random samples in [-u, u]^d using uniform
    X = np.random.uniform(-u, u, (n,d))

    #compute the dot product a^T X and assign lables based on the sign (neg or pos)
    y = np.sign(X @ a)

    #split testing and training data 70% for training and 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    return X_train, X_test, y_train, y_test, X, y, a

#generate 2d data set with n=100 samples
d = 2
n = 100
u = 5
X_train, X_test, y_train, y_test, X, y, a = make_classification(d, n, u)

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
