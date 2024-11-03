import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data for linear regression
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
y = y.reshape(-1, 1)  # Reshape for compatibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Linear Regression Oracle with generated data
oracle = LinRegOracle(X_train, y_train)

# Parameters for optimization methods
initial_point = np.zeros(X.shape[1])
step_sizes = [0.01, 0.05, 0.1]  # Different step sizes for Gradient Descent

# Function to run Gradient Descent and Newton's Method with different step sizes
def run_experiment(oracle, x0, step_sizes, max_iter=100, tol=1e-6):
    results = {}

    # Test Gradient Descent with different step sizes
    for step_size in step_sizes:
        x = x0
        values = []
        for i in range(max_iter):
            values.append(oracle.func(x))
            x = x - step_size * oracle.grad(x)
            if np.linalg.norm(oracle.grad(x)) < tol:
                break
        results[f"Gradient Descent (step size={step_size})"] = values

    # Run Newton's Method
    x = x0
    values = []
    for i in range(max_iter):
        values.append(oracle.func(x))
        try:
            dx = np.linalg.solve(oracle.hess(x), -oracle.grad(x))
            x = x + dx
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dx) < tol:
            break
    results["Newton's Method"] = values

    # Plotting results
    plt.figure(figsize=(10, 6))
    for label, values in results.items():
        plt.plot(values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence of Gradient Descent and Newton's Method with Varying Step Sizes")
    plt.legend()
    plt.show()

# Run the experiment
run_experiment(oracle, initial_point, step_sizes)
