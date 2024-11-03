def analyze_performance(oracle, x0, optimal_x=None, max_iter=100, step_size=0.01, tol=1e-6):
    # Data structures to store results
    performance_data = {
        "Gradient Descent": {"function_values": [], "distances": [], "iterations": 0},
        "Newton's Method": {"function_values": [], "distances": [], "iterations": 0}
    }
    
    # Run Gradient Descent
    x = x0
    for i in range(max_iter):
        f_val = oracle.func(x)
        performance_data["Gradient Descent"]["function_values"].append(f_val)
        
        # Calculate distance to optimal solution if known
        if optimal_x is not None:
            distance = np.linalg.norm(x - optimal_x)
            performance_data["Gradient Descent"]["distances"].append(distance)
        
        # Update x with gradient step
        x = x - step_size * oracle.grad(x)
        
        # Stopping criterion
        if np.linalg.norm(oracle.grad(x)) < tol:
            performance_data["Gradient Descent"]["iterations"] = i + 1
            break

    # Run Newton's Method
    x = x0
    for i in range(max_iter):
        f_val = oracle.func(x)
        performance_data["Newton's Method"]["function_values"].append(f_val)
        
        # Calculate distance to optimal solution if known
        if optimal_x is not None:
            distance = np.linalg.norm(x - optimal_x)
            performance_data["Newton's Method"]["distances"].append(distance)
        
        # Update x with Newton's step
        try:
            dx = np.linalg.solve(oracle.hess(x), -oracle.grad(x))
        except np.linalg.LinAlgError:
            print("Hessian is singular or nearly singular, stopping Newton's Method.")
            break
        x = x + dx
        
        # Stopping criterion
        if np.linalg.norm(dx) < tol:
            performance_data["Newton's Method"]["iterations"] = i + 1
            break

    # Print Summary Results
    print("Performance Summary:")
    for method, data in performance_data.items():
        print(f"\n{method}:")
        print(f" - Converged in {data['iterations']} iterations.")
        print(f" - Final objective function value: {data['function_values'][-1]:.4f}")
        if optimal_x is not None:
            print(f" - Final distance to optimal solution: {data['distances'][-1]:.4f}")
    
    # Plot Convergence Graphs
    plt.figure(figsize=(14, 5))
    
    # Objective function values over iterations
    plt.subplot(1, 2, 1)
    plt.plot(performance_data["Gradient Descent"]["function_values"], label="Gradient Descent")
    plt.plot(performance_data["Newton's Method"]["function_values"], label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Function Convergence")
    plt.legend()

    # Distance to optimal solution over iterations, if known
    if optimal_x is not None:
        plt.subplot(1, 2, 2)
        plt.plot(performance_data["Gradient Descent"]["distances"], label="Gradient Descent")
        plt.plot(performance_data["Newton's Method"]["distances"], label="Newton's Method")
        plt.xlabel("Iteration")
        plt.ylabel("Distance to Optimal Solution")
        plt.title("Distance to Optimal Solution Convergence")
        plt.legend()

    plt.show()
