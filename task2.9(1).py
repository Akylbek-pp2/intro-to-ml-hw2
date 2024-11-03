def adam_optimization(oracle, x0, step_size=0.01, max_iter=1000, tol=1e-6, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    m = np.zeros_like(x)  # Initialize first moment vector
    v = np.zeros_like(x)  # Initialize second moment vector
    values = []
    
    for i in range(1, max_iter + 1):
        f_val = oracle.func(x)
        values.append(f_val)
        
        grad = oracle.grad(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** i)
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2 ** i)
        
        # Update parameters
        x = x - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Stopping criterion
        if np.linalg.norm(grad) < tol:
            break
    
    return x, values
