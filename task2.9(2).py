from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and standardize the Boston housing dataset
data = load_boston()
X, y = data.data, data.target
y = y.reshape(-1, 1)  # Reshape y to be a column vector
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a regularized Linear Regression Oracle with the training data
oracle = LinRegOracleWithReg(X_train, y_train, reg_coef=0.1)
initial_point = np.zeros(X.shape[1])

# Run each optimization method
# Gradient Descent
x_gd, values_gd = gradient_descent_with_momentum(oracle, initial_point, step_size=0.01, momentum=0.0)

# Gradient Descent with Momentum
x_gd_momentum, values_gd_momentum = gradient_descent_with_momentum(oracle, initial_point, step_size=0.01, momentum=0.9)

# Newton's Method
x_newton, values_newton = [], []
x = initial_point
for i in range(100):
    values_newton.append(oracle.func(x))
    try:
        dx = np.linalg.solve(oracle.hess(x), -oracle.grad(x))
        x = x + dx
    except np.linalg.LinAlgError:
        break
    if np.linalg.norm(dx) < 1e-6:
        break

# Adam Optimization
x_adam, values_adam = adam_optimization(oracle, initial_point, step_size=0.01)

# Plot the convergence results
plt.figure(figsize=(10, 6))
plt.plot(values_gd, label="Gradient Descent")
plt.plot(values_gd_momentum, label="Gradient Descent with Momentum")
plt.plot(values_newton, label="Newton's Method")
plt.plot(values_adam, label="Adam Optimization")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.title("Comparison of Optimization Methods on Boston Housing Dataset")
plt.legend()
plt.show()
