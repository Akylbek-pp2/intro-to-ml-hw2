import matplotlib.pyplot as plt

def compare_methods(oracle, x0, optimal_x=None, max_iter=100, step_size=0.01, tol=1e-6):
    results = {
        "gradient_descent": {"values": [], "distances": []},
        "newton": {"values": [], "distances": []}
    }
    
    # Gradient Descent
    x = x0
    for i in range(max_iter):
        f_val = oracle.func(x)
        results["gradient_descent"]["values"].append(f_val)
        if optimal_x is not None:
            distance = np.linalg.norm(x - optimal_x)
            results["gradient_descent"]["distances"].append(distance)
        x = x - step_size * oracle.grad(x)
        if np.linalg.norm(oracle.grad(x)) < tol:
            break

    # Newton's Method
    x = x0
    for i in range(max_iter):
        f_val = oracle.func(x)
        results["newton"]["values"].append(f_val)
        if optimal_x is not None:
            distance = np.linalg.norm(x - optimal_x)
            results["newton"]["distances"].append(distance)
        try:
            dx = np.linalg.solve(oracle.hess(x), -oracle.grad(x))
        except np.linalg.LinAlgError:
            break
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break

    # Plot results
    plt.figure(figsize=(14, 5))

    # Plot function values
    plt.subplot(1, 2, 1)
    plt.plot(results["gradient_descent"]["values"], label="Gradient Descent")
    plt.plot(results["newton"]["values"], label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence of Objective Function")
    plt.legend()

    # Plot distances to optimum, if known
    if optimal_x is not None:
        plt.subplot(1, 2, 2)
        plt.plot(results["gradient_descent"]["distances"], label="Gradient Descent")
        plt.plot(results["newton"]["distances"], label="Newton's Method")
        plt.xlabel("Iteration")
        plt.ylabel("Distance to Optimum")
        plt.title("Distance to Optimum")
        plt.legend()

    plt.show()
