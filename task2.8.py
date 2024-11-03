import pandas as pd
import matplotlib.pyplot as plt

# Example data: These would be filled with actual results from tasks 2.1-2.7
# Assuming lists storing final objective values and iterations taken for each method
summary_data = {
    "Method": ["Gradient Descent", "Gradient Descent with Momentum", "Newton's Method"],
    "Final Objective Value": [0.0012, 0.0010, 0.0005],
    "Iterations": [1000, 500, 10],
    "Convergence Rate": ["Slow", "Moderate", "Fast"],
    "Stability": ["Stable with small step size", "Stable with momentum", "Stable but costly"]
}

# Convert to DataFrame for easy display and manipulation
df_summary = pd.DataFrame(summary_data)

# Display the summary table
print("Final Summary of Optimization Methods:")
print(df_summary)

# Plotting the final objective values and iterations for visual comparison
plt.figure(figsize=(12, 5))

# Plot for Final Objective Values
plt.subplot(1, 2, 1)
plt.bar(df_summary["Method"], df_summary["Final Objective Value"], color=['blue', 'orange', 'green'])
plt.xlabel("Method")
plt.ylabel("Final Objective Value")
plt.title("Final Objective Value Comparison")

# Plot for Iterations
plt.subplot(1, 2, 2)
plt.bar(df_summary["Method"], df_summary["Iterations"], color=['blue', 'orange', 'green'])
plt.xlabel("Method")
plt.ylabel("Iterations Taken")
plt.title("Iterations Comparison")

plt.tight_layout()
plt.show()
