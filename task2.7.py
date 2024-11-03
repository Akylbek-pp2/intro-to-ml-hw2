import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class LinRegOracleWithReg(BaseSmoothOracle):
    """
    Oracle for regularized linear regression:
       func(x) = 1/m ||Ax - b||^2 + (reg_coef/2) * ||x||^2.
    """
    def __init__(self, A, b, reg_coef=1.0):
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.reg_coef = reg_coef

    def func(self, x):
        residuals = self.A @ x - self.b
        return np.sum(residuals ** 2) / self.m + (self.reg_coef / 2) * np.dot(x, x)

    def grad(self, x):
        return 2 * self.A.T @ (self.A @ x - self.b) / self.m + self.reg_coef * x

    def hess(self, x):
        return 2 * self.A.T @ self.A / self.m + self.reg_coef * np.eye(self.A.shape[1])

# Momentum-based Gradient Descent
def gradient_descent_with_momentum(oracle, x0, step_size=0.01, max_iter=1000, tol=1e-6, momentum=0.9):
    x = x0
    v = np.zeros_like(x)  # Initialize momentum
    values = []
    for i in range(max_iter):
        f_val = oracle.func(x)
        values.append(f_val)
        
        grad = oracle.grad(x)
        v = momentum * v - step_size * grad
        x = x + v
        
        if np.linalg.norm(grad) < tol:
            break
    return x, values

# Experiment with different settings
def run_experiments_with_regularization_and_momentum(X_train, y_train, x0, step_size=0.01, max_iter=100, reg_coef=0.1, momentum=0.9):
    oracle = LinRegOracleWithReg(X_train, y_train, reg_coef=reg_coef)
    
    # Gradient Descent with momentum
    x_gd_momentum, values_gd_momentum = gradient_descent_with_momentum(oracle, x0, step_size=step_size, max_iter=max_iter, momentum=momentum)
    
    # Standard Gradient Descent for comparison
    x_gd_standard, values_gd_standard = gradient_descent_with_momentum(oracle, x0, step_size=step_size, max_iter=max_iter, momentum=0)

    # Newton's Method for comparison
    x_newton, values_newton = [], []
    x = x0
    for i in range(max_iter):
        f_val = oracle.func(x)
        values_newton.append(f_val)
        try:
            dx = np.linalg.solve(oracle.hess(x), -oracle.grad(x))
            x = x + dx
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dx) < tol:
            break

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(values_gd_momentum, label="Gradient Descent with Momentum")
    plt.plot(values_gd_standard, label="Standard Gradient Descent")
    plt.plot(values_newton, label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence with Regularization and Momentum")
    plt.legend()
    plt.show()

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run experiments
initial_point = np.zeros(X.shape[1])
run_experiments_with_regularization_and_momentum(X_train, y_train, initial_point, step_size=0.01, max_iter=100, reg_coef=0.1, momentum=0.9)
