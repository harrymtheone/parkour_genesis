import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

# Parameters for the von Mises distribution
kappa = 4  # Concentration parameter
mu = 0     # Mean (location) parameter

# Create an array of angles (in radians) from -π to π
theta = np.linspace(-np.pi, np.pi, 1000)

# Compute the PDF of the von Mises distribution
pdf_values = vonmises.pdf(theta, kappa, loc=mu)

# Plot the von Mises distribution
plt.figure(figsize=(8, 6))
plt.plot(theta, pdf_values, label=f"κ = {kappa}, μ = {mu}", color='blue')
plt.title("Von Mises Distribution")
plt.xlabel("Angle (radians)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.show()