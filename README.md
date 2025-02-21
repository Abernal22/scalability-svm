# Assignment 2: Scability of Support Vector Machines

# Purposes 
- Understanding the most popular Support Vector Machines (SVMs).
- Designing a basic algorithm to generate linearly separable datasets.
-  Understanding and implementing the basic linear SVM with soft margin.
-  Understanding and using the built-in SVMs in scikit-learn.
- Evaluating the scalability of SVMs on linear separable datasets.

## Implementation Tasks

### 1. (7 points) mplement a Python class named LinearSVC which learns a linear Support Vector Classifier (SVC)from a set of training data. The class is required to have the following components:

- (2 points) A constructor init which initialize an SVC using the given learning rate, number of
epochs and a random seed. (Similar to the perceptron class in our textbook.)
- (3 points) A training function fit which trains the SVC based on a given labeled dataset. We consider
the soft-margin SVC using a hinge loss. You are required to integrate L2-regularization and expose the
regularization parameter to users.
- (1 point) A function net input which computes the preactivation value for a given input sample.
- (1 point) A function predict which generates the prediction for a given input sample.

### 2. (3 points) Write a Python function make classification which generates a set of linearly separable databased on a random separation hyperplane. We learned that an (d − 1)-dimensional hyperplane can be defined as the set of points in the Euclidean space Rd satisfying an equation  ̄aT  ̄x = b, i.e., { ̄x ∈ Rd |  ̄aT  ̄x = b}. For simplicity, we assume that b = 0, then the hyperplane can be determined by a random vector  ̄a. We use this idea to design the following algorithm to generate random data which are linearly separable:

- Step 1. Randomly generate a d-dimensional vector  ̄a.
- Step 2. Randomly select n samples  ̄x1, . . . ,  ̄xn in the range of [−u, u] in each dimension. You may use
a uniform or Gaussian distribution to do so.
- Step 3. Give each  ̄xi a label yi such that if  ̄aT  ̄x < 0 then yi = −1, otherwise yi = 1

### Therefore, your function should have the following parameters that should given by the user: d, n, u, and a random seed for reproducing the data. You need to additionally subdivide the dataset to a training dataset (70%) and a test dataset (30%). You may use the scikit-learn function to do so, but make sure that you specify the random seed such that the subdivision is reproducible.

### 3. (4 points) Investigate the scalability of the LinearSVC class you have implemented. You may consider the datasets of the combinations of the following scales: d = 10, 50, 100, 500, 1000 and n = 500, 1000, 5000, 10000, 100000. Please feel free to adjust the scales according to your computers’ configurations, however the time costs should be obviously different. Make sure that you use the same dataset for each combination. This can be controlled by using the same random seed (see textbook).

### 4. (6 points) Read the scikit-learn documentation for SVMs. We are going to investigate the performance of solving primal and dual problems in linear classification. You may use the same datasets generated in the previous task. The easiest way to reuse a dataset is to keep all data in a file. In this task, you should use the built-in class LinearSVC in sklearn.svm. You may use the hinge loss and the default value for the regularization parameter. For each scale combination, e.g., d = 500, n = 1000, compare the time costs and prediction accuracies (on the test dataset) of training a linear SVC by solving the primal and dual problems respectively.