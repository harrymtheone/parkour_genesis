import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulate Z_θ(s): 5 quantiles (τ = [0.2, 0.4, 0.6, 0.8, 1.0])
quantile_fractions = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
quantile_values = np.sort(np.random.uniform(0, 10, 5))  # Predicted quantile values

# Each quantile has equal probability mass (1/N)
prob_mass = np.ones(5) / 5

# Plot PDF and CDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# PDF: Discrete distribution over quantile values
ax1.bar(quantile_values, prob_mass, width=0.5, alpha=0.7, edgecolor='k')
ax1.set_title('Probability Density Function (PDF)\nof Value Distribution $Z_θ(s)$')
ax1.set_xlabel('Value (Return)')
ax1.set_ylabel('Probability Mass')
ax1.grid(alpha=0.3)

# CDF: Step function from quantiles
cdf_values = np.cumsum(prob_mass)
ax2.step(quantile_values, cdf_values, where='post', label='CDF')
ax2.scatter(quantile_values, cdf_values, color='red', zorder=5)
ax2.set_title('Cumulative Distribution Function (CDF)')
ax2.set_xlabel('Value (Return)')
ax2.set_ylabel('Cumulative Probability')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()