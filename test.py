import numpy as np
import matplotlib.pyplot as plt

# Define the interval
y = np.logspace(np.log10(0.0001), np.log10(0.05), 1000)

# Original log-uniform interval
a = np.log(0.0001)
b = np.log(0.05)

# Compute PDF: 1 / ((b - a) * y)
pdf = 1 / ((b - a) * y)

# Plot with log scale on x-axis
plt.figure(figsize=(8, 4))
plt.plot(y, pdf, label='PDF of log-uniform distribution', color='blue')
plt.xscale('log')
plt.xlabel('y (log scale)')
plt.ylabel('Density')
plt.title('PDF of y = exp(x), where x ~ Uniform(log(0.0001), log(0.05))')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
