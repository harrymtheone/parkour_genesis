import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises

# Create an array of angles (in radians) from -π to π
theta = np.linspace(0., 1., 1000)

# Compute the PDF of the von Mises distribution
swing1 = vonmises.cdf(2 * np.pi * theta, 50, loc=0.2 * 2 * np.pi)
swing2 = vonmises.cdf(2 * np.pi * theta, 50, loc=0.6 * 2 * np.pi)
swing = swing1 * (1 - swing2)

stance1 = vonmises.cdf(2 * np.pi * theta, 50, loc=0.6 * 2 * np.pi)
stance2 = vonmises.cdf(2 * np.pi * theta, 50, loc=0.2 * 2 * np.pi)
stance = stance1 * (1 - stance2)

c = stance

# Plot the von Mises distribution
plt.figure(figsize=(8, 6))
plt.plot(theta, c, color='blue')
plt.title("Von Mises Distribution")
plt.xlabel("Angle (radians)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.show()
